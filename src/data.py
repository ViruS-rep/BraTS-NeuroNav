"""Patch dataset with ET-aware sampling and augmentation."""
import random
import numpy as np
import torch
from torch.utils.data import Dataset

from config import CACHE_DIR, ROI


def _pick_center(labels):
    """35% centred on an ET voxel, 40% on any tumour voxel, 25% anywhere."""
    r = random.random()
    idx = None
    if r < 0.35:
        idx = np.flatnonzero(labels == 3)
    elif r < 0.75:
        idx = np.flatnonzero(labels > 0)

    if idx is not None and idx.size > 0:
        flat = idx[random.randrange(idx.size)]
        center = np.unravel_index(flat, labels.shape)
    else:
        center = tuple(random.randrange(s) for s in labels.shape)

    # small jitter so the tumour is not always dead-centre
    return [int(c) + random.randint(-16, 16) for c in center]


def _augment(im, lab):
    for axis in (1, 2, 3):                       # flips along D, H, W
        if random.random() < 0.5:
            im = np.flip(im, axis=axis)
            lab = np.flip(lab, axis=axis - 1)
    im = np.ascontiguousarray(im)
    lab = np.ascontiguousarray(lab)
    for c in range(im.shape[0]):                 # per-channel intensity
        if random.random() < 0.5:
            im[c] = im[c] * random.uniform(0.9, 1.1) + random.uniform(-0.1, 0.1)
    return im, lab


class BratsPatchDataset(Dataset):
    def __init__(self, volume_ids, samples_per_volume=2, augment=True, roi=ROI):
        self.ids = list(volume_ids)
        self.n = samples_per_volume
        self.augment = augment
        self.roi = roi

    def __len__(self):
        return len(self.ids) * self.n

    def __getitem__(self, index):
        vid = self.ids[index % len(self.ids)]
        images = np.load(CACHE_DIR / f"volume_{vid}_images.npy", mmap_mode="r")
        labels = np.load(CACHE_DIR / f"volume_{vid}_labels.npy")

        center = _pick_center(labels)
        starts = [
            int(np.clip(c - r // 2, 0, s - r))
            for c, r, s in zip(center, self.roi, labels.shape)
        ]
        sl = tuple(slice(st, st + r) for st, r in zip(starts, self.roi))

        im = np.asarray(images[(slice(None),) + sl], dtype=np.float32)
        lab = np.asarray(labels[sl], dtype=np.int64)

        if self.augment:
            im, lab = _augment(im, lab)

        img_t = torch.from_numpy(np.ascontiguousarray(im))
        lab_t = torch.from_numpy(np.ascontiguousarray(lab))
        if self.augment and random.random() < 0.15:
            img_t = img_t + 0.05 * torch.randn_like(img_t)
        return img_t, lab_t