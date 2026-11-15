"""Paths, constants and verdict thresholds in one place."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"

PROCESSED_CSV = PROCESSED_DIR / "clean.csv"
MODEL_PATH = MODELS_DIR / "model.joblib"
METRICS_PATH = MODELS_DIR / "metrics.json"

RANDOM_SEED = 42

# Verdict bands on ratio = asking_price / predicted_price.
# Defaults from the brief; to be tuned on validation residuals in train.py.
GOOD_DEAL_MAX = 0.85
FAIR_MAX = 1.15
OVERPRICED_MAX = 1.40
SUSPICIOUS_LOW = 0.60
