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

# Features
MIN_LISTINGS_PER_LOCALITY = 3  # rarer localities are grouped into "other"
OTHER_LOCALITY = "other"

# Split
VAL_SIZE, TEST_SIZE = 0.15, 0.15
CV_FOLDS = 5

# Verdict bands on ratio = asking_price / predicted_price.
# train.py turns these percentiles of the VALIDATION ratios (actual / predicted) into
# thresholds, rounded to the nearest 0.05, and saves them with the model:
#   below SUSPICIOUS low pct        -> Suspicious (far too cheap: typo, scam, missing info)
#   SUSPICIOUS low .. FAIR low pct  -> Good Deal
#   FAIR low .. FAIR high pct       -> Fairly Priced   (80% of real listings)
#   FAIR high .. SUSPICIOUS high    -> Overpriced
#   above SUSPICIOUS high pct       -> Suspicious (far above normal market spread)
FAIR_PCT = (10, 90)
SUSPICIOUS_PCT = (2.5, 97.5)
THRESHOLD_ROUNDING = 0.05
# The fair-price range shown to users is predicted x [fair low, fair high] threshold,
# so "Fairly Priced" means exactly "asking price is inside the shown range".
