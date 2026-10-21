
from pathlib import Path
import h5py
import numpy as np

DATA_DIR = Path(
    r"C:\Users\Viraj\Downloads\BraTS2020_training_data\content\data"
)

# Inspect several slices from different volumes.
samples = [
    (1, 50), (1, 96), (9, 50), (9, 96),
    (25, 75), (50, 100), (100, 80), (200, 60),
]

channel_totals = np.zeros(3, dtype=np.int64)
overlap_totals = np.zeros((3, 3), dtype=np.int64)

for volume_id, slice_id in samples:
    path = DATA_DIR / f"volume_{volume_id}_slice_{slice_id}.h5"

    if not path.exists():
        print("Missing:", path.name)
        continue

    with h5py.File(path, "r") as f:
        image = f["image"][:]
        mask = f["mask"][:]

    sums = mask.sum(axis=(0, 1)).astype(int)
    channel_totals += sums

    print(f"\n{path.name}")
    print("Image shape:", image.shape)
    print("Mask positive pixels per channel:", sums.tolist())
    print("Image channel means:",
          np.round(image.mean(axis=(0, 1)), 3).tolist())

    for i in range(3):
        for j in range(3):
            overlap_totals[i, j] += np.logical_and(
                mask[..., i] > 0,
                mask[..., j] > 0
            ).sum()

print("\nTotal positive pixels across inspected slices:")
print(channel_totals.tolist())

print("\nPairwise positive-pixel overlaps:")
print(overlap_totals)
