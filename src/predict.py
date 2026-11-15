"""Load the trained model, predict a fair price, assign a verdict, and explain the estimate.

Used by the Streamlit app. Quick check from the command line:  python src/predict.py
"""
from dataclasses import dataclass, field
from functools import lru_cache

import joblib
import numpy as np
import pandas as pd

from config import MODEL_PATH, OTHER_LOCALITY

GOOD_DEAL, FAIR, OVERPRICED, SUSPICIOUS = "Good Deal", "Fairly Priced", "Overpriced", "Suspicious"
VERDICTS = [GOOD_DEAL, FAIR, OVERPRICED, SUSPICIOUS]

# Sensible input limits (roughly what the model saw in training).
LIMITS = {"area_sqft": (300, 30000), "bhk": (1, 10), "bathrooms": (1, 12), "balconies": (0, 3)}
FRIENDLY = {"area_sqft": "Area (sq ft)", "bhk": "BHK", "bathrooms": "Bathrooms", "balconies": "Balconies"}


# ---------------------------------------------------------------- formatting

def fmt_price(lakhs: float) -> str:
    """₹ lakhs below 1 crore, ₹ crores above (100 lakhs = 1 crore)."""
    if lakhs >= 100:
        return f"₹{lakhs / 100:.2f} Cr"
    return f"₹{lakhs:.1f}L" if lakhs < 10 else f"₹{lakhs:.0f}L"


# ---------------------------------------------------------------- verdict

def verdict_for_ratio(ratio: float, th: dict) -> str:
    """ratio = asking price / estimated fair price. Thresholds come from train.py (validation data)."""
    if ratio < th["suspicious_low"] or ratio > th["suspicious_high"]:
        return SUSPICIOUS
    if ratio < th["fair_low"]:
        return GOOD_DEAL
    if ratio <= th["fair_high"]:
        return FAIR
    return OVERPRICED


def verdict_reason(verdict: str, ratio: float) -> str:
    pct = abs(ratio - 1) * 100
    above_below = "above" if ratio > 1 else "below"
    return {
        GOOD_DEAL: f"Asking price is {pct:.0f}% below our estimate, cheaper than most similar listings.",
        FAIR: f"Asking price is within the normal range for similar listings ({pct:.0f}% {above_below} our estimate).",
        OVERPRICED: f"Asking price is {pct:.0f}% above our estimate, higher than most similar listings.",
        SUSPICIOUS: (f"Asking price is {pct:.0f}% {above_below} our estimate, far outside normal market spread. "
                     "Check for a typo, missing details, or a listing that is too good to be true."),
    }[verdict]


# ---------------------------------------------------------------- model loading

@lru_cache(maxsize=1)
def load_bundle() -> dict:
    """Load the model + metadata once, and build the SHAP explainer once."""
    import shap  # imported here so the rest of the module stays light

    bundle = joblib.load(MODEL_PATH)
    pipeline = bundle["pipeline"].regressor_          # the fitted Pipeline inside the log-target wrapper
    bundle["prep"] = pipeline.named_steps["prep"]
    bundle["explainer"] = shap.TreeExplainer(pipeline.named_steps["model"])
    return bundle


# ---------------------------------------------------------------- input checks

def validate(listing: dict, asking_lakhs: float) -> tuple[list[str], list[str]]:
    """Return (errors, warnings). Errors block a prediction; warnings are shown alongside it."""
    errors, warnings = [], []
    for key, (lo, hi) in LIMITS.items():
        value = listing.get(key)
        if value is None or not lo <= value <= hi:
            errors.append(f"{FRIENDLY[key]} should be between {lo:,} and {hi:,}.")
    if asking_lakhs is None or asking_lakhs <= 0:
        errors.append("Asking price must be a positive number.")
    if errors:
        return errors, warnings
    if listing["area_sqft"] / listing["bhk"] < 300:
        warnings.append("Under 300 sq ft per bedroom is unusual; the estimate may be unreliable.")
    if listing["bathrooms"] > listing["bhk"] + 2:
        warnings.append("Many more bathrooms than bedrooms is unusual; the estimate may be unreliable.")
    return errors, warnings


# ---------------------------------------------------------------- explanations

def _describe(feature: str, row: dict, typed_location: str) -> str:
    """Plain-language name for a factor, using the listing's own value."""
    generic = " as a generic locality" if row["location"] == OTHER_LOCALITY else ""
    return {
        "location": f"Locality ({typed_location}{generic})",
        "area_sqft": f"Area ({row['area_sqft']:,.0f} sq ft)",
        "bhk": f"{row['bhk']} BHK",
        "bathrooms": f"{row['bathrooms']} bathroom{'s' if row['bathrooms'] != 1 else ''}",
        "balconies": f"{row['balconies']} balcon{'ies' if row['balconies'] != 1 else 'y'}",
        "area_type": f"Area type ({row['area_type']})",
        "ready_to_move": "Ready to move in" if row["ready_to_move"] else "Under construction",
    }[feature]


def explain(X: pd.DataFrame, estimate: float, bundle: dict, typed_location: str, top_n: int = 5) -> list[dict]:
    """Top factors pushing the estimate up or down, in rupees.

    SHAP splits log(estimate) into base value + one contribution per feature (exactly additive).
    A contribution phi in log space means "this factor multiplies the price by e^phi", so its
    rupee effect is estimate * (1 - e^-phi): how much lower/higher the estimate would be if this
    factor were at a typical value instead.
    """
    Xt = bundle["prep"].transform(X)
    names = bundle["prep"].get_feature_names_out()
    phi = pd.Series(bundle["explainer"](Xt).values[0], index=names)
    # The area-type 0/1 columns belong to one human-level factor, so add them up.
    phi = phi.groupby(lambda n: "area_type" if n.startswith("area_type_") else n).sum()

    row = X.iloc[0].to_dict()
    factors = []
    for feature, p in phi.items():
        rupees = estimate * (1 - np.exp(-p))
        label = _describe(feature, row, typed_location)
        factors.append({
            "feature": feature,
            "label": label,
            "effect_lakhs": float(rupees),
            "text": f"{label}: {'raises' if rupees > 0 else 'lowers'} "
                    f"the estimate by about {fmt_price(abs(rupees))}",
        })
    factors.sort(key=lambda f: abs(f["effect_lakhs"]), reverse=True)
    # Skip negligible factors (< ₹0.5L) so the list stays meaningful.
    return [f for f in factors[:top_n] if abs(f["effect_lakhs"]) >= 0.5]


# ---------------------------------------------------------------- main entry point

@dataclass
class Result:
    estimate_lakhs: float
    range_lakhs: tuple[float, float]
    asking_lakhs: float
    ratio: float
    verdict: str
    reason: str
    factors: list[dict]
    typical_lakhs: float                 # SHAP base value: the model's "typical listing" price
    notes: list[str] = field(default_factory=list)


def predict(listing: dict, asking_lakhs: float) -> Result:
    """listing: location, area_type, area_sqft, bhk, bathrooms, balconies, ready_to_move (0/1).
    Raises ValueError with friendly messages if the inputs are invalid."""
    errors, notes = validate(listing, asking_lakhs)
    if errors:
        raise ValueError(" ".join(errors))
    bundle = load_bundle()

    row = dict(listing)
    if row["location"] not in bundle["localities"]:
        notes.append(f"'{row['location']}' has too few listings in our data, so it is treated as a "
                     "generic Bengaluru locality. The estimate is less specific.")
        row["location"] = OTHER_LOCALITY
    if row["area_type"] not in bundle["area_types"]:
        notes.append(f"Unknown area type '{row['area_type']}'; using the most common type instead.")
        row["area_type"] = "Super built-up"
    X = pd.DataFrame([row])[bundle["features"]]

    estimate = float(bundle["pipeline"].predict(X)[0])
    th = bundle["thresholds"]
    ratio = asking_lakhs / estimate
    verdict = verdict_for_ratio(ratio, th)
    return Result(
        estimate_lakhs=estimate,
        range_lakhs=(estimate * th["fair_low"], estimate * th["fair_high"]),
        asking_lakhs=asking_lakhs,
        ratio=ratio,
        verdict=verdict,
        reason=verdict_reason(verdict, ratio),
        factors=explain(X, estimate, bundle, typed_location=listing["location"]),
        typical_lakhs=float(np.exp(bundle["explainer"].expected_value[0])),
        notes=notes,
    )


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")  # Windows consoles default to a codepage without ₹
    example = {"location": "Whitefield", "area_type": "Super built-up", "area_sqft": 1200,
               "bhk": 2, "bathrooms": 2, "balconies": 1, "ready_to_move": 1}
    r = predict(example, asking_lakhs=70)
    print(f"Estimate {fmt_price(r.estimate_lakhs)} (range {fmt_price(r.range_lakhs[0])} to "
          f"{fmt_price(r.range_lakhs[1])}) -> {r.verdict}: {r.reason}")
    for f in r.factors:
        print("  -", f["text"])
