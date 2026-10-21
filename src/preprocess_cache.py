"""Build a NORMALISED cache from the raw HDF5 slices.

Per volume:
  1. stack the 155 slices -> images (4,D,H,W), labels (D,H,W)
  2. check the 3 mask channels are mutually exclusive (abort if not)
  3. brain mask = voxels that differ from the background value
  4. per channel: clip to [0.5, 99.5] percentile of brain voxels,
     z-score using brain voxels only, background set to 0
  5. save float16 images + uint8 labels

Label map: 0=background, 1=NCR/NET, 2=edema, 3=enhancing tumour.
Run:  python preprocess_cache.py
"""
import h5py
import numpy as np
from tqdm import tqdm

from config import (RAW_DIR, CACHE_DIR, NUM_VOLUMES, DEPTH, HEIGHT, WIDTH,
                    CHANNELS)


def load_raw_volume(volume_id):
    images, masks = [], []
    for d in range(DEPTH):
        path = RAW_DIR / f"volume_{volume_id}_slice_{d}.h5"
        with h5py.File(path, "r") as f:
            images.append(f["image"][:])
            masks.append(f["mask"][:])
    images = np.stack(images).astype(np.float32)          # (D,H,W,C)
    images = np.transpose(images, (3, 0, 1, 2))           # (C,D,H,W)
    masks = np.stack(masks).astype(np.uint8)              # (D,H,W,3)
    return images, masks


def masks_to_labels(masks, volume_id):
    positive = masks > 0
    if positive.sum(axis=-1).max() > 1:
        raise RuntimeError(
            f"Volume {volume_id}: mask channels overlap, so the "
            "c0 + 2*c1 + 3*c2 label map is invalid. Fix before training."
        )
    labels = (positive[..., 0].astype(np.uint8)
              + 2 * positive[..., 1].astype(np.uint8)
              + 3 * positive[..., 2].astype(np.uint8))
    return labels


def normalise(images, volume_id):
    out = np.zeros_like(images, dtype=np.float32)
    brain = np.zeros(images.shape[1:], dtype=bool)

    for c in range(images.shape[0]):
        x = images[c]
        # Background value estimated from the image border region.
        bg = np.median(x[:, :10, :10])
        brain |= np.abs(x - bg) > 1e-4

    if brain.sum() == 0:
        raise RuntimeError(f"Volume {volume_id}: empty brain mask")

    for c in range(images.shape[0]):
        vals = images[c][brain]
        lo, hi = np.percentile(vals, [0.5, 99.5])
        vals = np.clip(vals, lo, hi)
        mean, std = vals.mean(), vals.std()
        out[c][brain] = (vals - mean) / (std + 1e-8)

    return out, float(brain.mean())


def atomic_save(path, array):
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        np.save(f, array)
    tmp.replace(path)


def main():
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    print("Source:", RAW_DIR)
    print("Cache :", CACHE_DIR)

    for volume_id in tqdm(range(1, NUM_VOLUMES + 1)):
        img_path = CACHE_DIR / f"volume_{volume_id}_images.npy"
        lab_path = CACHE_DIR / f"volume_{volume_id}_labels.npy"
        if img_path.exists() and lab_path.exists():
            continue

        images, masks = load_raw_volume(volume_id)
        labels = masks_to_labels(masks, volume_id)
        images, brain_frac = normalise(images, volume_id)

        assert images.shape == (CHANNELS, DEPTH, HEIGHT, WIDTH)
        assert labels.shape == (DEPTH, HEIGHT, WIDTH)
        if not (0.02 < brain_frac < 0.6):
            print(f"\nWARNING volume {volume_id}: brain fraction {brain_frac:.3f}")

        atomic_save(img_path, images.astype(np.float16))
        atomic_save(lab_path, labels)

    print("\nDone. Run verify_cache.py next.")


if __name__ == "__main__":
    main()