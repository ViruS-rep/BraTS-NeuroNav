"""Central configuration. Edit paths here only."""
from pathlib import Path
import torch

# ---- paths -----------------------------------------------------------
RAW_DIR = Path(r"C:\Users\Viraj\Downloads\BraTS2020_training_data\content\data")
CACHE_DIR = Path(r"D:\BraTS_cache_v2")          # NEW cache (normalised, float16)
CKPT_DIR = Path("checkpoints")
OUT_DIR = Path("outputs")
SPLIT_PATH = CKPT_DIR / "brats_clean_split.json"  # SAME split file as before
BEST_PATH = CKPT_DIR / "segresnet_best.pth"
LAST_PATH = CKPT_DIR / "segresnet_last.pth"
HISTORY_PATH = OUT_DIR / "training_history.json"

# ---- data geometry ---------------------------------------------------
NUM_VOLUMES = 369
DEPTH, HEIGHT, WIDTH, CHANNELS = 155, 240, 240, 4

# ---- device ----------------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_CUDA = DEVICE == "cuda"
SEED = 42

# ---- training (tuned for a 4 GB GPU) ---------------------------------
ROI = (96, 96, 96)         # if you hit CUDA OOM: (80, 80, 80)
BATCH_SIZE = 1
SAMPLES_PER_VOLUME = 2     # patches drawn per volume per epoch
NUM_WORKERS = 4            # lower to 2 if RAM is tight
EPOCHS = 100
LR = 2e-4
WEIGHT_DECAY = 1e-5

# ---- validation / inference -----------------------------------------
VAL_EVERY = 5              # epochs between full-volume validations
VAL_SUBSET = 20            # validation volumes used during training
VAL_OVERLAP = 0.25         # fast during training
TEST_OVERLAP = 0.5         # accurate for final evaluation
SW_BATCH_SIZE = 1
ET_MIN_VOXELS = 300        # post-processing threshold for enhancing tumour

REGION_NAMES = ("TC", "WT", "ET")   # order of the 3 output channels