"""Paths, constants and verdict thresholds in one place."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"

RAW_CSV = RAW_DIR / "Bengaluru_House_Data.csv"
PROCESSED_CSV = PROCESSED_DIR / "clean.csv"
MODEL_PATH = MODELS_DIR / "model.joblib"
METRICS_PATH = MODELS_DIR / "metrics.json"

RANDOM_SEED = 42

# Cleaning rules (see src/clean.py)
MAX_BHK = 10
MIN_SQFT_PER_BHK = 300
MAX_BATH_OVER_BHK = 2          # drop if bathrooms > bhk + this
PPSF_PCT_LOW, PPSF_PCT_HIGH = 0.01, 0.99

# Verdict bands on ratio = asking_price / predicted_price.
# Defaults from the brief; to be tuned on validation residuals in train.py.
GOOD_DEAL_MAX = 0.85
FAIR_MAX = 1.15
OVERPRICED_MAX = 1.40
SUSPICIOUS_LOW = 0.60
