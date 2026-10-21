
import torch
import monai
import nibabel as nib

print("PyTorch:", torch.__version__)
print("PyTorch CUDA build:", torch.version.cuda)
print("MONAI:", monai.__version__)
print("Nibabel:", nib.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print(
        "GPU memory:",
        round(
            torch.cuda.get_device_properties(0).total_memory
            / 1024**3,
            2,
        ),
        "GB",
    )

    x = torch.randn(1024, 1024, device="cuda")
    y = x @ x
    torch.cuda.synchronize()

    print("GPU computation successful:", y.device)
else:
    print("GPU not detected. Check your PyTorch installation.")
