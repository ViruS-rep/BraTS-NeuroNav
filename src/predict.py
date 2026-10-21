"""Visualise prediction vs ground truth for one volume.
  python predict.py --volume 5            # picks the slice with most tumour
  python predict.py --volume 5 --slice 80
"""
import argparse
import matplotlib.pyplot as plt
import numpy as np
import torch

from config import *
from infer import load_cached, predict_regions
from metrics import postprocess, regions_to_labels
from model import build_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--volume", type=int, required=True)
    ap.add_argument("--slice", type=int, default=None)
    ap.add_argument("--ckpt", default=str(BEST_PATH))
    args = ap.parse_args()

    model = build_model().to(DEVICE)
    model.load_state_dict(torch.load(args.ckpt, map_location=DEVICE, weights_only=True))

    images, labels = load_cached(args.volume)
    pred = regions_to_labels(postprocess(predict_regions(model, images)))

    z = args.slice if args.slice is not None else int((labels > 0).sum(axis=(1, 2)).argmax())
    fig, ax = plt.subplots(1, 3, figsize=(14, 5))
    ax[0].imshow(images[1, z], cmap="gray"); ax[0].set_title(f"T1ce (slice {z})")
    ax[1].imshow(labels[z], cmap="viridis", vmin=0, vmax=3); ax[1].set_title("Ground truth")
    ax[2].imshow(pred[z], cmap="viridis", vmin=0, vmax=3); ax[2].set_title("Prediction")
    for a in ax:
        a.axis("off")
    plt.tight_layout()
    OUT_DIR.mkdir(exist_ok=True)
    out = OUT_DIR / f"prediction_volume_{args.volume}_slice_{z}.png"
    plt.savefig(out, dpi=150)
    print("Saved:", out)
    plt.show()


if __name__ == "__main__":
    main()