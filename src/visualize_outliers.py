
import numpy as np
import matplotlib.pyplot as plt

data = np.load("data/processed/volume_1.npz")
images = data["image"]       # (4, D, H, W)
labels = data["label"]       # (D, H, W)

slice_id = 50
for i, name in enumerate(["T1", "T1ce", "T2", "FLAIR"]):
    x = images[i, slice_id]
    print(
        name,
        "min:", float(x.min()),
        "max:", float(x.max()),
        "std:", float(x.std()),
        "percentiles:",
        np.percentile(x, [1, 50, 99, 99.5]).round(3)
    )

print("Label counts:", np.bincount(labels[slice_id].ravel(), minlength=4))
names = ["T1", "T1ce", "T2", "FLAIR"]

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

for i, name in enumerate(names):
    ax = axes.flat[i]
    image_slice = images[i, slice_id]

    # Robust display range; does not modify training data.
    low, high = np.percentile(image_slice, [1, 99.5])
    ax.imshow(image_slice, cmap="gray", vmin=low, vmax=high)
    ax.set_title(name)
    ax.axis("off")

ax = axes.flat[4]
ax.imshow(labels[slice_id], cmap="viridis", vmin=0, vmax=3)
ax.set_title("Segmentation labels")
ax.axis("off")

ax = axes.flat[5]
combined = np.max(images[:, slice_id], axis=0)
ax.imshow(combined > 20, cmap="gray")
ax.set_title("Locations where any modality > 20")
ax.axis("off")

plt.tight_layout()
plt.savefig("outputs/outlier_slice_138.png", dpi=150)
plt.show()
