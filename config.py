"""Central configuration for the 300W HRNet facial landmark project.

Keep paths, training hyperparameters, evaluation settings, and alignment settings here.
Training/evaluation scripts should import from this file instead of hard-coding paths.
"""
from pathlib import Path
import torch


# -------------------------------
# Project directories
# -------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
TEST_DIR = PROJECT_ROOT / "test"
CACHE_ROOT = PROJECT_ROOT / "cache"


# -------------------------------
# Dataset / preprocessing settings
# -------------------------------
DATASET_PATH = "hub://activeloop/300w"

IMG_SIZE = 256
HEATMAP_SIZE = 64
NUM_LANDMARKS = 68

CROP_SCALE = 1.5
HEATMAP_SIGMA = 2

USE_DISK_CACHE = True
CACHE_DIR = CACHE_ROOT / "processed_300w"


# -------------------------------
# Device settings
# -------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# -------------------------------
# Training settings
# -------------------------------
BATCH_SIZE = 8
EPOCHS = 200
LR = 1e-3
WEIGHT_DECAY = 0.0
SAVE_EVERY_EPOCHS = 10

RESUME = False
RESUME_PATH = CHECKPOINT_DIR / "hrnet_epoch_200.pth"

SAVE_PATH = CHECKPOINT_DIR / "landmark_HRNet.pth"
BEST_PATH = CHECKPOINT_DIR / "best_epoch.pth"
LOSS_LOG_PATH = CHECKPOINT_DIR / "HRNet_loss_log.csv"


# -------------------------------
# Model settings
# -------------------------------
HRNET_WIDTH = 18  # HRNet-W18. Change to 32/48 only if the model factory supports it.
PRETRAINED_PATH = ""  # Empty string means random initialization.
FINAL_CONV_KERNEL = 1
BN_MOMENTUM = 0.01


# -------------------------------
# Heatmap decoding settings
# -------------------------------
USE_QUARTER_OFFSET = True


# -------------------------------
# Evaluation settings
# -------------------------------
# EVAL_CHECKPOINT_PATH = CHECKPOINT_DIR / "hrnet_epoch_200.pth"
EVAL_CHECKPOINT_PATH = CHECKPOINT_DIR / "best_epoch.pth"
EVAL_OUT_DIR = TEST_DIR / "full_eval_epoch_200_fixed"
EVAL_BATCH_SIZE = 16

# NME normalization options:
# - "inter_ocular": outer eye corner distance, gt[36] to gt[45]; preferred for 300W-style reporting.
# - "inter_eye": left/right eye-center distance; this was used in the original code.
# - "bbox_diag": face landmark bounding-box diagonal.
NME_NORM_TYPE = "inter_ocular"
FAILURE_THRESHOLD = 0.08
AUC_THRESHOLD = 0.08


# -------------------------------
# Alignment settings
# -------------------------------
ALIGN_CHECKPOINT_PATH = EVAL_CHECKPOINT_PATH
ALIGN_OUT_DIR = TEST_DIR / "single_predict_fixed"
ALIGN_OUTPUT_SIZE = 112
ALIGN_INDEX_START = 30
ALIGN_INDEX_END = 40  # Python range end; evaluates [start, end).

# ArcFace-style 5-point template defined for 112 x 112 aligned faces.
ALIGN_TEMPLATE_112 = [
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041],
]