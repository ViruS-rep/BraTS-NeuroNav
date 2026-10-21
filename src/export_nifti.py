"""Export FLAIR / prediction / ground truth as NIfTI.

  python export_nifti.py --volume 300
  python export_nifti.py --volume 300 --reference path\\to\\BraTS20_Training_300_flair.nii.gz

--reference copies the real affine/header from an original BraTS file.
Labels are saved in BraTS convention (ET = 4).
The h5 slice axes may be transposed relative to the original NIfTI; check the
overlay in 3D Slicer and add --swap-xy if the result is mirrored/rotated.
"""
import argparse
import nibabel as nib
import numpy as np
import torch

from config import *
from infer import load_cached, predict_regions
from metrics import postprocess, regions_to_labels
from model import build_model


def to_xyz(arr_dhw, swap_xy):
    out = np.transpose(arr_dhw, (1, 2, 0))        # (D,H,W) -> (H,W,D)
    return np.transpose(out, (1, 0, 2)) if swap_xy else out


def to_brats(labels):
    out = labels.astype(np.uint8).copy()
    out[labels == 3] = 4
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--volume", type=int, default=300)
    ap.add_argument("--ckpt", default=str(BEST_PATH))
    ap.add_argument("--reference", default=None)
    ap.add_argument("--swap-xy", action="store_true")
    args = ap.parse_args()

    model = build_model().to(DEVICE)
    model.load_state_dict(torch.load(args.ckpt, map_location=DEVICE, weights_only=True))
    images, true_labels = load_cached(args.volume)
    pred = regions_to_labels(postprocess(predict_regions(model, images)))

    affine, header = np.eye(4), None
    if args.reference:
        ref = nib.load(args.reference)
        affine, header = ref.affine, ref.header
    else:
        print("WARNING: identity affine used (no --reference given).")

    OUT = OUT_DIR / "nifti"; OUT.mkdir(parents=True, exist_ok=True)
    v = args.volume
    items = {
        f"volume_{v}_flair.nii.gz": to_xyz(images[3], args.swap_xy).astype(np.float32),
        f"volume_{v}_prediction.nii.gz": to_xyz(to_brats(pred), args.swap_xy),
        f"volume_{v}_ground_truth.nii.gz": to_xyz(to_brats(true_labels), args.swap_xy),
    }
    for name, arr in items.items():
        nib.save(nib.Nifti1Image(arr, affine, header), str(OUT / name))
        print("Saved:", OUT / name)


if __name__ == "__main__":
    main()