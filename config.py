"""
SegFusionFormer
Centralized Project Configuration

M.Tech Computer Science and Engineering
Karunya Institute of Technology and Sciences
"""

from pathlib import Path
import os


# ============================================
# 1. PROJECT ROOT
# ============================================

PROJECT_ROOT = Path(__file__).resolve().parent


# ============================================
# 2. DATASET CONFIGURATION
# ============================================

# Allows the dataset to be stored outside GitHub.
# Set SEGFUSION_DATA_DIR to override the default.

DATASET_DIR = Path(
    os.environ.get(
        "SEGFUSION_DATA_DIR",
        str(PROJECT_ROOT / "dataset")
    )
).expanduser().resolve()

IMAGE_DIR = DATASET_DIR / "images"

# NIH ChestX-ray14 metadata
METADATA_CSV = DATASET_DIR / "Data_Entry_2017.csv"

# Support alternative NIH metadata filename
if not METADATA_CSV.exists():
    alternative_csv = (
        DATASET_DIR / "Data_Entry_2017_v2020.csv"
    )

    if alternative_csv.exists():
        METADATA_CSV = alternative_csv

BBOX_CSV = DATASET_DIR / "BBox_List_2017.csv"


# ============================================
# 3. LUNG SEGMENTATION DATASET
# ============================================

SEGMENTATION_DATA_DIR = (
    PROJECT_ROOT / "segmentation_dataset"
)


# ============================================
# 4. MODEL AND OUTPUT PATHS
# ============================================

MODELS_DIR = PROJECT_ROOT / "models"

OUTPUT_DIR = PROJECT_ROOT / "outputs"

RESULTS_DIR = PROJECT_ROOT / "results"


# ============================================
# 5. PROJECT INFORMATION
# ============================================

PROJECT_NAME = "SegFusionFormer"

NUM_CLASSES = 14

CLASSIFICATION_IMAGE_SIZE = 224

SEGMENTATION_IMAGE_SIZE = 256
