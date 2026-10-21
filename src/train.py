"""Single training script (replaces train / train_backup / train_clean /
train_improved).

  python train.py            # fresh run
  python train.py --resume   # continue from checkpoints/segresnet_last.pth

The split file is loaded, never overwritten. Test volumes are never read.
"""
import argparse
import json
import random
import time

import numpy as np
import torch
import torch.nn as nn
from monai.losses import DiceLoss
from torch.utils.data import DataLoader

from config import *
from data import BratsPatchDataset
from infer import load_cached, predict_regions
from metrics import labels_to_regions, postprocess, region_scores
from model import build_model


def load_or_create_split():
    if SPLIT_PATH.exists():
        split = json.loads(SPLIT_PATH.read_text(encoding="utf-8"))
        print("Loaded existing split:", SPLIT_PATH)
    else:
        print("WARNING: split file missing, creating it (same algorithm as before).")
        ids = list(range(1, NUM_VOLUMES + 1))
        random.Random(SEED).shuffle(ids)
        n_train, n_val = round(0.70 * len(ids)), round(0.15 * len(ids))
        split = {"seed": SEED,
                 "train": sorted(ids[:n_train]),
                 "validation": sorted(ids[n_train:n_train + n_val]),
                 "test": sorted(ids[n_train + n_val:])}
        SPLIT_PATH.parent.mkdir(parents=True, exist_ok=True)
        SPLIT_PATH.write_text(json.dumps(split, indent=2), encoding="utf-8")

    tr, va, te = map(set, (split["train"], split["validation"], split["test"]))
    if tr & va or tr & te or va & te:
        raise RuntimeError("Split overlap detected.")
    print(f"train={len(tr)} validation={len(va)} test={len(te)} (test not used here)")
    return split


def validate(model, val_ids):
    scores = {f"dice_{n}": [] for n in REGION_NAMES}
    for vid in val_ids:
        images, labels = load_cached(vid)
        pred = predict_regions(model, images, overlap=VAL_OVERLAP)
        pred = postprocess(pred)
        s = region_scores(pred, labels)
        for k, v in s.items():
            scores[k].append(v)
    means = {k: float(np.mean(v)) for k, v in scores.items()}
    means["mean_dice"] = float(np.mean(list(means.values())))
    return means


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    if USE_CUDA:
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cudnn.benchmark = True
        print("GPU:", torch.cuda.get_device_name(0))

    CKPT_DIR.mkdir(exist_ok=True); OUT_DIR.mkdir(exist_ok=True)
    split = load_or_create_split()
    val_ids = split["validation"][:VAL_SUBSET]

    train_ds = BratsPatchDataset(split["train"], SAMPLES_PER_VOLUME, augment=True)
    loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, persistent_workers=NUM_WORKERS > 0,
        pin_memory=USE_CUDA, drop_last=True,
    )

    model = build_model().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda e: (1 - e / EPOCHS) ** 0.9)       # poly decay
    scaler = torch.amp.GradScaler("cuda", enabled=USE_CUDA)
    dice_loss = DiceLoss(sigmoid=True, squared_pred=True, batch=True)
    bce_loss = nn.BCEWithLogitsLoss()

    start_epoch, best, history = 1, -1.0, []
    if args.resume and LAST_PATH.exists():
        ck = torch.load(LAST_PATH, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ck["model"]); optimizer.load_state_dict(ck["optimizer"])
        scheduler.load_state_dict(ck["scheduler"]); scaler.load_state_dict(ck["scaler"])
        start_epoch, best, history = ck["epoch"] + 1, ck["best"], ck["history"]
        print(f"Resumed from epoch {ck['epoch']} (best={best:.4f})")

    for epoch in range(start_epoch, EPOCHS + 1):
        model.train()
        t0, total = time.time(), 0.0
        for step, (images, labels) in enumerate(loader, 1):
            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)
            target = labels_to_regions(labels)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(DEVICE, dtype=torch.float16, enabled=USE_CUDA):
                logits = model(images)
            logits = logits.float()
            loss = dice_loss(logits, target) + bce_loss(logits, target)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer); scaler.update()
            total += loss.item()

            if step % 100 == 0:
                print(f"  epoch {epoch} step {step}/{len(loader)} loss={loss.item():.4f}",
                      flush=True)
        scheduler.step()

        record = {"epoch": epoch, "train_loss": total / len(loader),
                  "lr": scheduler.get_last_lr()[0], "minutes": (time.time() - t0) / 60}
        msg = (f"Epoch {epoch}/{EPOCHS} | loss={record['train_loss']:.4f} | "
               f"lr={record['lr']:.2e} | {record['minutes']:.1f} min")

        if epoch % VAL_EVERY == 0 or epoch == EPOCHS:
            val = validate(model, val_ids)
            record.update(val)
            msg += (f" | VAL TC={val['dice_TC']:.4f} WT={val['dice_WT']:.4f} "
                    f"ET={val['dice_ET']:.4f} mean={val['mean_dice']:.4f}")
            if val["mean_dice"] > best:
                best = val["mean_dice"]
                torch.save(model.state_dict(), BEST_PATH)
                msg += "  <- new best"
        print(msg, flush=True)

        history.append(record)
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
                    "epoch": epoch, "best": best, "history": history}, LAST_PATH)
        HISTORY_PATH.write_text(json.dumps(history, indent=2), encoding="utf-8")

    print(f"\nDone. Best validation mean Dice (TC/WT/ET): {best:.4f}")
    print("Best weights:", BEST_PATH)
    print("Next: python evaluate.py --split validation")


if __name__ == "__main__":
    main()