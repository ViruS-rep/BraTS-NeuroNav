
from pathlib import Path

import h5py
import numpy as np

DATA_DIR = Path(
    r"C:\Users\Viraj\Downloads\BraTS2020_training_data\content\data"
)

VOLUME_ID = 1
NUM_SLICES = 155


def load_volume(volume_id):
    images = []
    masks = []

    for slice_id in range(NUM_SLICES):
        path = DATA_DIR / (
            f"volume_{volume_id}_slice_{slice_id}.h5"
        )

        if not path.exists():
            raise FileNotFoundError(path)

        with h5py.File(path, "r") as f:
            images.append(f["image"][:].astype(np.float32))
            masks.append(f["mask"][:].astype(np.uint8))

    # (D, H, W, C)
    images = np.stack(images, axis=0)
    masks = np.stack(masks, axis=0)

    # Convert image to (C, D, H, W) for PyTorch.
    images = np.transpose(images, (3, 0, 1, 2))

    # Convert the three mutually exclusive masks into
    # one integer label map: 0=background, 1, 2, 3=tumor classes.
    label_map = (
        masks[..., 0].astype(np.uint8)
        + 2 * masks[..., 1].astype(np.uint8)
        + 3 * masks[..., 2].astype(np.uint8)
    )

    # Convert labels from (D, H, W) to (D, H, W);
    # PyTorch training will add a batch dimension later.
    return images, label_map


if __name__ == "__main__":
    images, labels = load_volume(VOLUME_ID)

    print("Image shape (C, D, H, W):", images.shape)
    print("Label shape (D, H, W):", labels.shape)
    print("Image dtype:", images.dtype)
    print("Label dtype:", labels.dtype)
    print("Image range:", images.min(), "to", images.max())
    print("Unique labels:", np.unique(labels))
    print("Voxel counts per class:")

    for class_id in range(4):
        count = int(np.sum(labels == class_id))
        print(f"  Class {class_id}: {count:,}")

    # Save only this test volume for now.
    output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        output_dir / f"volume_{VOLUME_ID}.npz",
        image=images,
        label=labels,
    )

    print("Saved test volume successfully.")
