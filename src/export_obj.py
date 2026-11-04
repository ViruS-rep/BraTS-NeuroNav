
import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
import trimesh
from skimage.measure import marching_cubes


# Saved NIfTI uses BraTS labels:
# 0 = background, 1 = NCR/NET, 2 = edema, 4 = enhancing tumour
LABELS = {
    1: "necrotic_non_enhancing",
    2: "edema",
    4: "enhancing_tumor",
}


def export_mask(binary_mask, affine, output_path, name):
    binary_mask = binary_mask.astype(np.uint8)

    if not np.any(binary_mask):
        print(f"Skipping {name}: no voxels found")
        return

    # Padding closes surfaces touching the volume boundaries.
    padded = np.pad(binary_mask, 1, mode="constant")
    vertices, faces, _, _ = marching_cubes(padded, level=0.5)
    vertices -= 1  # Undo padding offset.

    # Convert voxel coordinates to NIfTI world coordinates.
    vertices_world = nib.affines.apply_affine(affine, vertices)

    mesh = trimesh.Trimesh(
        vertices=vertices_world,
        faces=faces,
        process=True,
    )
    mesh.export(str(output_path))

    print(
        f"Saved {output_path} | "
        f"vertices={len(mesh.vertices)} | faces={len(mesh.faces)}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", required=True, help="Prediction .nii or .nii.gz"
    )
    parser.add_argument("--output", default="outputs/obj")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = (
        Path(args.output)
        / input_path.name.replace(".nii.gz", "").replace(".nii", "")
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    nii = nib.load(str(input_path))
    mask = np.asarray(nii.dataobj)

    print("Input shape:", mask.shape)
    print("Labels and counts:", np.unique(mask, return_counts=True))
    print("Affine:\n", nii.affine)

    # Individual tumour subregions.
    for label, name in LABELS.items():
        export_mask(
            mask == label,
            nii.affine,
            output_dir / f"{name}.obj",
            name,
        )

    # Whole tumour = union of all tumour subregions.
    export_mask(
        mask > 0,
        nii.affine,
        output_dir / "whole_tumor.obj",
        "whole_tumor",
    )


if __name__ == "__main__":
    main()
