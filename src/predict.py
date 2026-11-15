"""Load the trained model, predict a fair price, assign a verdict, and explain the estimate.

Phase 3 only needs the verdict rule (train.py uses it to check band shares).
Prediction and explanations are added in phase 4.
"""

GOOD_DEAL, FAIR, OVERPRICED, SUSPICIOUS = "Good Deal", "Fairly Priced", "Overpriced", "Suspicious"
VERDICTS = [GOOD_DEAL, FAIR, OVERPRICED, SUSPICIOUS]


def verdict_for_ratio(ratio: float, th: dict) -> str:
    """ratio = asking price / estimated fair price. Thresholds come from train.py (validation data)."""
    if ratio < th["suspicious_low"] or ratio > th["suspicious_high"]:
        return SUSPICIOUS
    if ratio < th["fair_low"]:
        return GOOD_DEAL
    if ratio <= th["fair_high"]:
        return FAIR
    return OVERPRICED
