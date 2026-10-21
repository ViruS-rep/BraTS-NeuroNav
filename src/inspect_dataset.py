
from pathlib import Path
import re
import h5py
import numpy as np
import matplotlib.pyplot as plt

DATA_DIR = Path(
    r"C:\Users\Viraj\Downloads\BraTS2020_training_data\content\data"
)

# Find HDF5 files and inspect their volume/slice naming.
files = list(DATA_DIR.glob("*.h5"))
print("HDF5 files found:", len(files))

pattern = re.compile(r"volume_(\d+)_slice_(\d+)\.h5$")
volume_slices = {}

for path in files:
    match = pattern.match(path.name)
    if match:
        volume_id, slice_id = map(int, match.groups())
        volume_slices.setdefault(volume_id, set()).add(slice_id)

print("Volumes found:", len(volume_slices))

for volume_id in sorted(volume_slices)[:10]:
    slices = sorted(volume_slices[volume_id])
    missing = sorted(set(range(slices[0], slices[-1] + 1)) - set(slices))
    print(
        f"Volume {volume_id}: {len(slices)} slices, "
        f"range {slices[0]}-{slices[-1]}, "
        f"missing slices: {missing[:10]}"
    )

# Inspect and display one sample.
sample_path = DATA_DIR / "volume_9_slice_96.h5"

with h5py.File(sample_path, "r") as f:
    image = f["image"][:]
    mask = f["mask"][:]

print("\nSample:", sample_path.name)
print("Image shape:", image.shape)
print("Mask shape:", mask.shape)

for c in range(image.shape[-1]):
    print(
        f"Image channel {c}: "
        f"min={image[..., c].min():.3f}, "
        f"max={image[..., c].max():.3f}"
    )

for c in range(mask.shape[-1]):
    values, counts = np.unique(mask[..., c], return_counts=True)
    print(f"Mask channel {c} values/counts:", dict(zip(values.tolist(), counts.tolist())))

fig, axes = plt.subplots(2, 4, figsize=(14, 7))

for c in range(4):
    axes[0, c].imshow(image[..., c], cmap="gray")
    axes[0, c].set_title(f"Image channel {c}")
    axes[0, c].axis("off")

for c in range(3):
    axes[1, c].imshow(mask[..., c], cmap="gray", vmin=0, vmax=1)
    axes[1, c].set_title(f"Mask channel {c}")
    axes[1, c].axis("off")

axes[1, 3].imshow(image[..., 0], cmap="gray")
axes[1, 3].imshow(
    np.max(mask, axis=-1),
    cmap="autumn",
    alpha=0.45,
    vmin=0,
    vmax=1,
)
axes[1, 3].set_title("Mask overlay")
axes[1, 3].axis("off")

plt.tight_layout()
plt.show()
