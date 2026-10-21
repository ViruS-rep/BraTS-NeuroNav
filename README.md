# BraTS-NeuroNav

3D brain tumor segmentation on **BraTS 2020** using a **SegResNet** (MONAI / PyTorch), designed to train and run on a **4 GB GPU**.

The model takes four MRI modalities (T1, T1ce, T2, FLAIR) and predicts the three standard BraTS tumor regions: **Tumor Core (TC)**, **Whole Tumor (WT)** and **Enhancing Tumor (ET)**.

> **Independent test Dice (TC / WT / ET): 0.836 mean** (56 held-out volumes)  
> Validation mean Dice: 0.845 (55 volumes)

> ⚠️ **Research use only.** This project is not a medical device and has not been clinically validated. Do not use it for diagnosis, treatment planning or surgical navigation.

---

## Results

Mean Dice (full volumes, sliding-window inference). Hausdorff95 was not computed for these runs.

| Region | Validation (55 volumes) | **Independent test (56 volumes)** |
|--------|------------------------|-----------------------------------|
| Tumor Core (TC) | 0.8369 | 0.8207 |
| Whole Tumor (WT) | 0.9053 | 0.9014 |
| Enhancing Tumor (ET) | 0.7932 | 0.7861 |
| **Mean across regions** | **0.8451** | **0.8361** |

The test split was held out from training and checkpoint selection, so the **test column is the number to report**. The validation numbers are slightly higher because checkpoints were selected on that split. The small gap (about 0.009) suggests the model is not overfitting to the validation set.

Evaluation uses full volumes and the BraTS convention for empty regions (see [Evaluation](#evaluation)). To add Hausdorff95, run `python evaluate.py --split validation --hd95`.

---

## Features

- **Leakage-safe split:** 70 / 15 / 15 train / validation / test, split **by patient** and saved to `checkpoints/brats_clean_split.json`. The training script never overwrites it and never reads test volumes.
- **Normalized cache:** per-channel percentile clipping (0.5–99.5) and z-scoring over brain voxels only, stored as float16 memmaps for fast loading.
- **Mask sanity check:** preprocessing aborts if the three mask channels overlap, which would corrupt the label map.
- **Region-based training:** three sigmoid outputs (TC, WT, ET) with Dice + BCE loss.
- **ET-aware patch sampling:** 96³ patches, 35% centered on enhancing tumor, 40% on any tumor, 25% random.
- **4 GB VRAM friendly:** mixed precision, batch size 1, GroupNorm, sliding-window inference with CPU-side stitching.
- **Resumable training** with a poly learning-rate schedule and gradient clipping.
- **BraTS-style metrics:** Dice, Hausdorff95, and small-ET post-processing.
- **Export:** NIfTI output with optional affine copied from an original BraTS file.

---

## Project structure

```
BraTS-NeuroNav/
├── config.py            # all paths and hyperparameters
├── preprocess_cache.py  # HDF5 slices -> normalized float16 volumes (one-time)
├── verify_cache.py      # sanity checks on the cache
├── model.py             # SegResNet (3 sigmoid outputs)
├── data.py              # patch dataset, ET-aware sampling, augmentation
├── metrics.py           # region conversion, Dice, HD95, post-processing
├── infer.py             # sliding-window inference
├── train.py             # training + periodic full-volume validation
├── evaluate.py          # full-volume evaluation (validation / test)
├── predict.py           # visualize prediction vs ground truth
├── export_nifti.py      # export FLAIR / prediction / GT as NIfTI
├── inspect_dataset.py   # (optional) browse the raw HDF5 data
├── inspect_h5.py        # (optional) inspect one HDF5 file
├── requirements.txt
└── .gitignore
```

Generated at runtime (git-ignored): `checkpoints/`, `outputs/`.

---

## Setup

### Requirements

- Python 3.10+
- NVIDIA GPU with **4 GB+** VRAM (CPU works but is very slow)
- About **26 GB** of free disk space for the cache
- 16 GB RAM recommended

### Install

```bash
git clone <your-repo-url>
cd BraTS-NeuroNav
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install torch --index-url https://download.pytorch.org/whl/cu121   # pick the build for your CUDA
pip install -r requirements.txt
```

### Data

This project expects the **BraTS 2020 training data in HDF5 slice format**: one file per slice, named `volume_<id>_slice_<k>.h5`, with:

- `image`: `(240, 240, 4)` → channels T1, T1ce, T2, FLAIR
- `mask`: `(240, 240, 3)` → mutually exclusive channels NCR/NET, edema, ET

369 volumes × 155 slices. BraTS data must be obtained under its own terms (see the [BraTS challenge](https://www.synapse.org/brats)); it is **not** included in this repository.

Edit the paths at the top of `config.py`:

```python
RAW_DIR   = Path(r"C:\path\to\BraTS2020_training_data\content\data")
CACHE_DIR = Path(r"D:\BraTS_cache_v2")
```

---

## Usage

Run in this order:

```bash
# 1. Build the normalized cache (one-time, slow)
python preprocess_cache.py

# 2. Verify the cache: expect mean ~0, std ~1, labels in [0,1,2,3], no NaN
python verify_cache.py

# 3. Check GPU memory for one training step (should be well under 4 GB)
python model.py

# 4. Train (resume an interrupted run with --resume)
python train.py
python train.py --resume

# 5. Evaluate on the validation split
python evaluate.py --split validation --hd95

# 6. Visualize a case
python predict.py --volume 5
python predict.py --volume 5 --slice 80

# 7. FINAL, run once: held-out test set
python evaluate.py --split test --confirm-test

# 8. Export NIfTI (optionally with the original affine)
python export_nifti.py --volume 300
python export_nifti.py --volume 300 --reference path/to/BraTS20_Training_300_flair.nii.gz
```

### Useful `evaluate.py` flags

| Flag | Meaning |
|------|---------|
| `--split {validation,test}` | which split to evaluate |
| `--ckpt PATH` | checkpoint (default: best) |
| `--overlap 0.5` | sliding-window overlap |
| `--hd95` | also compute Hausdorff95 (slower) |
| `--no-postprocess` | disable ET post-processing |
| `--confirm-test` | required to touch the test split |

---

## Method

### Labels and regions

Cached label map: `0` background, `1` NCR/NET, `2` edema, `3` enhancing tumor.

| Region | Definition |
|--------|-----------|
| TC | labels 1 + 3 |
| WT | labels 1 + 2 + 3 |
| ET | label 3 |

### Preprocessing

1. Stack the 155 slices into `(4, 155, 240, 240)`.
2. Verify the mask channels do not overlap, then build the label map.
3. Brain mask = voxels differing from the background value.
4. Per channel: clip to the 0.5–99.5 percentile of brain voxels, z-score over brain voxels, set background to 0.
5. Save float16 images and uint8 labels.

### Model

MONAI `SegResNet`: 4 input channels, 3 output channels, `init_filters=16`, `blocks_down=(1,2,2,4)`, `blocks_up=(1,1,1)`, dropout 0.2, GroupNorm.

### Training

| Setting | Value |
|---------|-------|
| Patch size | 96 × 96 × 96 |
| Batch size | 1 |
| Epochs | 100 (2 patches per volume per epoch) |
| Optimizer | AdamW, LR 2e-4, weight decay 1e-5 |
| Schedule | poly decay (power 0.9) |
| Loss | Dice (squared, sigmoid) + BCE-with-logits |
| Precision | mixed (fp16 autocast), gradient clipping at 1.0 |
| Augmentation | flips on all 3 axes, per-channel intensity scale/shift, Gaussian noise |
| Validation | 20 validation volumes every 5 epochs, full volume, sliding window |
| Model selection | best mean Dice (TC/WT/ET) on validation |

### Evaluation

- **Inference:** sliding window, ROI 96³, Gaussian blending, overlap 0.25 during training and 0.5 for final evaluation, threshold 0.5.
- **Post-processing:** if predicted ET is smaller than 300 voxels it is removed (those voxels stay in TC as NCR/NET), and ET ⊆ TC ⊆ WT is enforced.
- **Dice convention:** both empty → 1, exactly one empty → 0.
- **HD95 convention:** both empty → 0, exactly one empty → 373.13.

---

## Configuration

Everything lives in `config.py`. Common changes:

| Situation | Change |
|-----------|--------|
| CUDA out of memory | `ROI = (80, 80, 80)`; if still failing, `init_filters=8` in `model.py` |
| RAM pressure / slow startup on Windows | `NUM_WORKERS = 2` |
| Faster experiments | lower `EPOCHS`, raise `VAL_EVERY` |
| Stronger final numbers | raise `EPOCHS`, evaluate with `--overlap 0.5` |

---

## Limitations

- Trained and evaluated on BraTS 2020 only; no external validation.
- A single run and a single split. Results can vary by several points with a different seed or split.
- The HDF5 slice files do not store spatial metadata. Exports use an identity affine unless `--reference` is given, and the in-plane axes may be transposed relative to the original NIfTI. Verify orientation in 3D Slicer (use `--swap-xy` if needed) before any anatomical use.
- No test-time augmentation, ensembling or cross-validation.

## Ideas for future work

- 5-fold cross-validation and model ensembling
- Test-time augmentation (flips)
- Larger or deeper network (needs more VRAM) or nnU-Net comparison
- Largest-connected-component filtering for WT
- Orientation-correct export and a viewer integration for the navigation use case

---

## Acknowledgements and references

- B. Menze et al., "The Multimodal Brain Tumor Image Segmentation Benchmark (BRATS)", *IEEE TMI*, 2015.
- S. Bakas et al., "Advancing The Cancer Genome Atlas glioma MRI collections with expert segmentation labels and radiomic features", *Scientific Data*, 2017.
- S. Bakas et al., "Identifying the Best Machine Learning Algorithms for Brain Tumor Segmentation, Progression Assessment, and Overall Survival Prediction in the BRATS Challenge", arXiv:1811.02629, 2018.
- A. Myronenko, "3D MRI brain tumor segmentation using autoencoder regularization", 2018 (SegResNet).
- [MONAI](https://monai.io/): Medical Open Network for AI.

## License

Add your license here (for example MIT). The BraTS dataset is subject to its own terms of use and is not covered by this repository's license.
