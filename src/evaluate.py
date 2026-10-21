"""Full-volume evaluation (replaces evaluate / evaluate_clean /
evaluate_improved_validation).

  python evaluate.py --split validation
  python evaluate.py --split validation --hd95
  python evaluate.py --split test --confirm-test     # do this ONCE, at the end
"""
import argparse
import csv
import json

import numpy as np
import torch

from config import *
from infer import load_cached, predict_regions
from metrics import postprocess, region_scores
from model import build_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["validation", "test"], default="validation")
    ap.add_argument("--ckpt", default=str(BEST_PATH))
    ap.add_argument("--overlap", type=float, default=TEST_OVERLAP)
    ap.add_argument("--hd95", action="store_true", help="also compute Hausdorff95 (slower)")
    ap.add_argument("--no-postprocess", action="store_true")
    ap.add_argument("--confirm-test", action="store_true")
    args = ap.parse_args()

    if args.split == "test" and not args.confirm_test:
        raise SystemExit("Test set is reserved. Add --confirm-test to run it "
                         "(only after you have finished tuning).")

    split = json.loads(SPLIT_PATH.read_text(encoding="utf-8"))
    ids = split[args.split]
    model = build_model().to(DEVICE)
    model.load_state_dict(torch.load(args.ckpt, map_location=DEVICE, weights_only=True))
    model.eval()

    rows = []
    print(f"Evaluating {len(ids)} {args.split} volumes | ckpt={args.ckpt}")
    for i, vid in enumerate(ids, 1):
        images, labels = load_cached(vid)
        pred = predict_regions(model, images, overlap=args.overlap)
        if not args.no_postprocess:
            pred = postprocess(pred)
        s = region_scores(pred, labels, with_hd95=args.hd95)
        s["volume"] = vid
        rows.append(s)
        print(f"[{i}/{len(ids)}] vol {vid}: " +
              " ".join(f"{k}={v:.3f}" for k, v in s.items() if k != "volume"), flush=True)

    keys = [k for k in rows[0] if k != "volume"]
    print("\n" + "=" * 52)
    for k in keys:
        arr = np.array([r[k] for r in rows])
        print(f"{k:10s} mean={arr.mean():.4f}  std={arr.std():.4f}  median={np.median(arr):.4f}")
    dice_keys = [k for k in keys if k.startswith("dice")]
    print(f"Mean Dice (TC/WT/ET): {np.mean([np.mean([r[k] for r in rows]) for k in dice_keys]):.4f}")

    OUT_DIR.mkdir(exist_ok=True)
    out = OUT_DIR / f"eval_{args.split}.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["volume"] + keys)
        w.writeheader(); w.writerows(rows)
    print("Saved per-volume results:", out)


if __name__ == "__main__":
    main()