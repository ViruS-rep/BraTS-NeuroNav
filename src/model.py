"""SegResNet with 3 sigmoid outputs (TC, WT, ET). Sized for 4 GB VRAM."""
import torch
from monai.networks.nets import SegResNet


def build_model():
    return SegResNet(
        spatial_dims=3,
        in_channels=4,
        out_channels=3,            # TC, WT, ET (overlapping regions)
        init_filters=16,           # must be divisible by 8 (GroupNorm)
        blocks_down=(1, 2, 2, 4),
        blocks_up=(1, 1, 1),
        dropout_prob=0.2,
    )


if __name__ == "__main__":
    from config import DEVICE, ROI
    model = build_model().to(DEVICE)
    print("Parameters:", sum(p.numel() for p in model.parameters()))
    x = torch.randn(1, 4, *ROI, device=DEVICE)
    model.train()
    with torch.autocast(DEVICE, dtype=torch.float16, enabled=DEVICE == "cuda"):
        y = model(x)
    y.float().mean().backward()
    print("Output:", tuple(y.shape))
    if DEVICE == "cuda":
        print("Peak GPU MB (train step):",
              round(torch.cuda.max_memory_allocated() / 1024**2))