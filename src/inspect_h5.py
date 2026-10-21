
import h5py
import sys
import numpy as np

if len(sys.argv) < 2:
    print("Usage: python src\\inspect_h5.py <path-to-h5-file>")
    raise SystemExit(1)

file_path = sys.argv[1]

with h5py.File(file_path, "r") as f:
    print("File:", file_path)
    print("\nTop-level keys:", list(f.keys()))

    def inspect(name, obj):
        if isinstance(obj, h5py.Dataset):
            print(f"\nDataset: {name}")
            print("  Shape:", obj.shape)
            print("  Dtype:", obj.dtype)

            if obj.size > 0 and np.issubdtype(obj.dtype, np.number):
                sample = obj[()]
                print("  Min:", np.nanmin(sample))
                print("  Max:", np.nanmax(sample))
                print("  Unique values (sample):",
                      np.unique(sample)[:15])

    f.visititems(inspect)
