"""Region conversion, BraTS Dice / HD95, and ET post-processing."""
import numpy as np
import torch

from config import ET_MIN_VOXELS, REGION_NAMES


def labels_to_regions(lab):
    """torch (B,D,H,W) int labels -> (B,3,D,H,W) float [TC, WT, ET]."""
    tc = (lab == 1) | (lab == 3)
    wt = lab > 0
    et = lab == 3
    return torch.stack([tc, wt, et], dim=1).float()


def labels_to_regions_np(lab):
    """numpy (D,H,W) -> (3,D,H,W) bool [TC, WT, ET]."""
    return np.stack([(lab == 1) | (lab == 3), lab > 0, lab == 3])


def regions_to_labels(regions):
    """(3,D,H,W) bool -> label map for display (0/1/2/3)."""
    tc, wt, et = regions
    out = np.zeros(tc.shape, dtype=np.uint8)
    out[wt] = 2
    out[tc] = 1
    out[et] = 3
    return out


def postprocess(regions, et_min=ET_MIN_VOXELS):
    """Remove tiny ET blobs and enforce ET <= TC <= WT."""
    tc, wt, et = [r.copy() for r in regions]
    if et.sum() < et_min:
        et[:] = False                     # TC keeps those voxels as NCR/NET
    tc |= et
    wt |= tc
    return np.stack([tc, wt, et])


def dice(pred, true):
    """BraTS convention: both empty -> 1, one empty -> 0."""
    p, t = pred.sum(), true.sum()
    if p == 0 and t == 0:
        return 1.0
    if p == 0 or t == 0:
        return 0.0
    return float(2.0 * np.logical_and(pred, true).sum() / (p + t))


def hd95(pred, true):
    """BraTS convention: both empty -> 0, one empty -> 373.13."""
    p, t = pred.sum(), true.sum()
    if p == 0 and t == 0:
        return 0.0
    if p == 0 or t == 0:
        return 373.13
    from monai.metrics import compute_hausdorff_distance
    v = compute_hausdorff_distance(
        torch.from_numpy(pred)[None, None].float(),
        torch.from_numpy(true)[None, None].float(),
        include_background=True, percentile=95,
    )
    return float(v.item())


def region_scores(pred_regions, labels, with_hd95=False):
    """pred_regions (3,D,H,W) bool, labels (D,H,W) -> dict of scores."""
    true_regions = labels_to_regions_np(np.asarray(labels))
    out = {}
    for i, name in enumerate(REGION_NAMES):
        out[f"dice_{name}"] = dice(pred_regions[i], true_regions[i])
        if with_hd95:
            out[f"hd95_{name}"] = hd95(pred_regions[i], true_regions[i])
    return out