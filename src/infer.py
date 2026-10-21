"""Sliding-window inference on a full cached volume."""
import numpy as np
import torch
from monai.inferers import sliding_window_inference

from config import CACHE_DIR, DEVICE, USE_CUDA, ROI, SW_BATCH_SIZE


def load_cached(volume_id):
    images = np.load(CACHE_DIR / f"volume_{volume_id}_images.npy", mmap_mode="r")
    labels = np.load(CACHE_DIR / f"volume_{volume_id}_labels.npy")
    return np.asarray(images, dtype=np.float32), labels


@torch.no_grad()
def predict_regions(model, images, overlap=0.5, threshold=0.5):
    """images (4,D,H,W) float32 -> bool (3,D,H,W) [TC, WT, ET]."""
    model.eval()
    x = torch.from_numpy(images)[None]            # stays on CPU

    def predictor(t):
        with torch.autocast(DEVICE, dtype=torch.float16, enabled=USE_CUDA):
            return model(t).float()

    logits = sliding_window_inference(
        x, roi_size=ROI, sw_batch_size=SW_BATCH_SIZE, predictor=predictor,
        overlap=overlap, mode="gaussian",
        sw_device=DEVICE, device="cpu", progress=False,
    )
    return (torch.sigmoid(logits)[0] > threshold).numpy()