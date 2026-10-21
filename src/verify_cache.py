"""Sanity-check the cache. Replaces audit_masks / check_intensity /
inspect_outliers / visualize_outliers.  Run: python verify_cache.py"""
import numpy as np
from config import CACHE_DIR

IDS = [1, 9, 25, 50, 100, 200, 300, 369]
NAMES = ["T1", "T1ce", "T2", "FLAIR"]

for vid in IDS:
    img = np.load(CACHE_DIR / f"volume_{vid}_images.npy", mmap_mode="r")
    lab = np.load(CACHE_DIR / f"volume_{vid}_labels.npy")
    print(f"\nVolume {vid}: images {img.shape} {img.dtype}, "
          f"labels {lab.shape} {lab.dtype}")
    print("  label values:", np.unique(lab).tolist(),
          "| counts:", np.bincount(lab.ravel(), minlength=4).tolist())
    assert lab.max() <= 3, "label map invalid"

    for c, n in enumerate(NAMES):
        x = np.asarray(img[c], dtype=np.float32)
        brain = x != 0
        v = x[brain]
        print(f"  {n:5s} brain-voxels mean={v.mean():+.3f} std={v.std():.3f} "
              f"min={v.min():+.2f} max={v.max():+.2f} "
              f"nan={np.isnan(x).any()}")

print("\nExpect: mean ~0, std ~1, max <= ~4, no NaN, labels subset of [0,1,2,3].")