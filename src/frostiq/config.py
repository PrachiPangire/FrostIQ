"""Project-wide settings for FrostIQ."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"

# Experimental default only: the upper limit of the common 2-8 °C storage range.
# Every function that uses it accepts a different threshold.
DEFAULT_THRESHOLD_C = 8.0

DEFAULT_SEED = 42
