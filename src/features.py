"""Feature engineering shared by training and prediction.

Inputs the model sees: locality, area type, ready-to-move flag, area, BHK, bathrooms, balconies.
price_per_sqft is NOT an input: it is computed from the price we are trying to predict (leakage).
"""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import KFold
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler, TargetEncoder

from config import MIN_LISTINGS_PER_LOCALITY, OTHER_LOCALITY, RANDOM_SEED

CATEGORICAL = ["location", "area_type"]
NUMERIC = ["area_sqft", "bhk", "bathrooms", "balconies", "ready_to_move"]
FEATURES = CATEGORICAL + NUMERIC
TARGET = "price_lakhs"


def known_localities(train: pd.DataFrame, min_count: int = MIN_LISTINGS_PER_LOCALITY) -> list[str]:
    """Localities with enough training listings to learn a price level for. Fit on training data only."""
    counts = train["location"].value_counts()
    return sorted(counts[counts >= min_count].index)


def group_localities(df: pd.DataFrame, known: list[str]) -> pd.DataFrame:
    """Map rare or unseen localities to 'other' so the model always gets a category it knows."""
    df = df.copy()
    df["location"] = df["location"].where(df["location"].isin(known), OTHER_LOCALITY)
    return df


def tree_preprocessor() -> ColumnTransformer:
    """Locality -> one number: its typical (log) price level, learned from TRAINING rows only.
    TargetEncoder computes this with internal cross-fitting, so a training row never sees its own
    price, and it shrinks small localities toward the city average. Area type -> 0/1 columns.
    Numbers pass through unchanged (trees don't need scaling)."""
    return ColumnTransformer(
        [
            ("location", TargetEncoder(cv=KFold(5, shuffle=True, random_state=RANDOM_SEED)), ["location"]),
            ("area_type", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["area_type"]),
        ],
        remainder="passthrough",
        verbose_feature_names_out=False,
    )


def linear_preprocessor() -> ColumnTransformer:
    """The linear model also gets log(area), since price grows with area multiplicatively."""
    numeric = [c for c in NUMERIC if c != "area_sqft"]
    return ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
            ("log_area", FunctionTransformer(np.log), ["area_sqft"]),
            ("num", StandardScaler(), numeric),
        ]
    )
