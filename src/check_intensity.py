
import numpy as np

data = np.load("data/processed/volume_1.npz")
images = data["image"]

channel_names = ["T1", "T1ce", "T2", "FLAIR"]

for i, name in enumerate(channel_names):
    x = images[i]
    print(f"\n{name}")
    print(f"  Min:       {x.min():.4f}")
    print(f"  Max:       {x.max():.4f}")
    print(f"  Mean:      {x.mean():.4f}")
    print(f"  Std:       {x.std():.4f}")
    print(f"  Percentile 1:  {np.percentile(x, 1):.4f}")
    print(f"  Percentile 50: {np.percentile(x, 50):.4f}")
    print(f"  Percentile 99: {np.percentile(x, 99):.4f}")
    print(f"  Percentile 99.9: {np.percentile(x, 99.9):.4f}")
