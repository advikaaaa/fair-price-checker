"""Train the fair-price model, compare it with a linear baseline, and save everything predict.py needs.

Run:  python src/train.py
Outputs: models/model.joblib (pipeline + metadata) and models/metrics.json
"""
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.pipeline import Pipeline

import clean
from config import (
    CV_FOLDS,
    FAIR_PCT,
    METRICS_PATH,
    MODEL_PATH,
    PROCESSED_CSV,
    RANDOM_SEED,
    RAW_CSV,
    SUSPICIOUS_PCT,
    TEST_SIZE,
    THRESHOLD_ROUNDING,
    VAL_SIZE,
)
from features import FEATURES, TARGET, group_localities, known_localities, linear_preprocessor, tree_preprocessor
from predict import VERDICTS, verdict_for_ratio


def log_target(regressor):
    """Train on log(price) and convert predictions back to lakhs automatically.
    Prices are very skewed, so errors on log scale treat a ₹5L miss on a ₹40L flat
    and a ₹50L miss on a ₹4 Cr villa as similar mistakes."""
    return TransformedTargetRegressor(regressor=regressor, func=np.log, inverse_func=np.exp)


def build_models() -> dict:
    return {
        "Linear regression (baseline)": Pipeline([("prep", linear_preprocessor()), ("model", LinearRegression())]),
        "Random forest": Pipeline([("prep", tree_preprocessor()), ("model", RandomForestRegressor(
            n_estimators=200, min_samples_leaf=3, max_features=0.5, n_jobs=-1, random_state=RANDOM_SEED))]),
        "Gradient boosting": Pipeline([("prep", tree_preprocessor()), ("model", GradientBoostingRegressor(
            # Huber loss: less pulled around by the badly mispriced listings that exist in real data.
            loss="huber", n_estimators=600, learning_rate=0.05, max_depth=4, subsample=0.8,
            random_state=RANDOM_SEED))]),
    }


# Gradient boosting is the main model. On cross-validation it is roughly tied with the linear baseline:
# slightly better on MAPE and MAE (percentage error is what the verdict uses), slightly worse on RMSE/R².
# It also captures interactions (e.g. plots vs flats in pricey areas), is small on disk, and SHAP can
# explain it exactly. Random forest is kept for comparison only (similar accuracy, ~30x larger file).
MAIN_MODEL = "Gradient boosting"


def cross_validate(dev: pd.DataFrame) -> dict:
    """Average metrics over K folds. One test split of ~1,800 noisy listings can favour either
    model by luck (a few ₹10 Cr+ homes swing RMSE and R²), so this is the fairer comparison."""
    scores = {name: [] for name in build_models()}
    for fit_idx, eval_idx in KFold(CV_FOLDS, shuffle=True, random_state=RANDOM_SEED).split(dev):
        fit_part, eval_part = dev.iloc[fit_idx], dev.iloc[eval_idx]
        localities = known_localities(fit_part)
        fit_part, eval_part = group_localities(fit_part, localities), group_localities(eval_part, localities)
        for name, pipe in build_models().items():
            m = log_target(pipe).fit(fit_part[FEATURES], fit_part[TARGET])
            scores[name].append(evaluate(eval_part[TARGET], m.predict(eval_part[FEATURES])))
    return {name: pd.DataFrame(s).mean().to_dict() for name, s in scores.items()}


def evaluate(y_true, y_pred) -> dict:
    return {
        "MAE_lakhs": mean_absolute_error(y_true, y_pred),
        "RMSE_lakhs": mean_squared_error(y_true, y_pred) ** 0.5,
        "R2": r2_score(y_true, y_pred),
        "MAPE_pct": 100 * mean_absolute_percentage_error(y_true, y_pred),
    }


def tune_thresholds(val_ratio: np.ndarray) -> dict:
    """Turn the percentile rule in config.py into ratio thresholds, rounded to a readable step."""
    def pct(p):
        return round(round(np.percentile(val_ratio, p) / THRESHOLD_ROUNDING) * THRESHOLD_ROUNDING, 2)
    return {"suspicious_low": pct(SUSPICIOUS_PCT[0]), "fair_low": pct(FAIR_PCT[0]),
            "fair_high": pct(FAIR_PCT[1]), "suspicious_high": pct(SUSPICIOUS_PCT[1])}


def verdict_band_shares(ratio: np.ndarray, thresholds: dict) -> dict:
    """What share of real listings lands in each verdict band (ratio = actual / predicted)."""
    verdicts = pd.Series([verdict_for_ratio(r, thresholds) for r in ratio])
    return {v: float((verdicts == v).mean()) for v in VERDICTS}


def main():
    if not PROCESSED_CSV.exists():
        print("No cleaned data found, running cleaning first.\n")
        clean.main() if RAW_CSV.exists() else exit("Put the raw CSV in data/raw/ first.")
    df = pd.read_csv(PROCESSED_CSV)

    # 70 / 15 / 15 split with a fixed seed so results are reproducible.
    train, rest = train_test_split(df, test_size=VAL_SIZE + TEST_SIZE, random_state=RANDOM_SEED)
    val, test = train_test_split(rest, test_size=TEST_SIZE / (VAL_SIZE + TEST_SIZE), random_state=RANDOM_SEED)

    # Locality list is learned from training data only, then applied to every split.
    localities = known_localities(train)
    train_raw, val_raw = train, val
    train, val, test = (group_localities(d, localities) for d in (train, val, test))
    print(f"Rows: train {len(train):,} | val {len(val):,} | test {len(test):,}")
    print(f"Known localities: {len(localities)} (the rest are grouped as 'other', "
          f"{(train.location == 'other').mean():.0%} of training rows)\n")

    print(f"Running {CV_FOLDS}-fold cross-validation on train+val (test set untouched)...")
    cv = cross_validate(pd.concat([train_raw, val_raw]))
    results, fitted = {}, {}
    for name, pipe in build_models().items():
        model = log_target(pipe).fit(train[FEATURES], train[TARGET])
        fitted[name] = model
        results[name] = {"validation": evaluate(val[TARGET], model.predict(val[FEATURES])),
                         "test": evaluate(test[TARGET], model.predict(test[FEATURES])),
                         "cv_mean": cv[name]}
    model = fitted[MAIN_MODEL]

    for label, key in [(f"{CV_FOLDS}-fold cross-validation, mean", "cv_mean"), ("Held-out test set", "test")]:
        table = pd.DataFrame({n: r[key] for n, r in results.items()}).T.round(3)
        print(f"\n{label} (prices in Rs lakhs):")
        print(table.to_string())
    print(f"\nMain model: {MAIN_MODEL}")

    # Residual ratios on validation data drive the fair-price range and the verdict bands.
    val_ratio = val[TARGET].to_numpy() / model.predict(val[FEATURES])
    test_ratio = test[TARGET].to_numpy() / model.predict(test[FEATURES])
    pcts = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    ratio_pcts = {f"p{p}": float(np.percentile(val_ratio, p)) for p in pcts}
    print("\nValidation actual/predicted ratio percentiles:",
          {k: round(v, 2) for k, v in ratio_pcts.items()})
    thresholds = tune_thresholds(val_ratio)
    print("Verdict thresholds (asking / estimate):", thresholds)
    shares = {"validation": verdict_band_shares(val_ratio, thresholds),
              "test": verdict_band_shares(test_ratio, thresholds)}
    for split, s in shares.items():
        print(f"Verdict shares on real {split} listings:", {k: f"{v:.0%}" for k, v in s.items()})

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "pipeline": model,
        "model_name": MAIN_MODEL,
        "features": FEATURES,
        "localities": localities,
        "area_types": sorted(train["area_type"].unique()),
        "ratio_percentiles": ratio_pcts,
        "thresholds": thresholds,
        "test_metrics": results[MAIN_MODEL]["test"],
        "n_train": len(train),
    }, MODEL_PATH, compress=3)
    METRICS_PATH.write_text(json.dumps({
        "main_model": MAIN_MODEL,
        "metrics": results,
        "validation_ratio_percentiles": ratio_pcts,
        "verdict_thresholds": thresholds,
        "verdict_shares": shares,
    }, indent=2))
    size_mb = MODEL_PATH.stat().st_size / 1e6
    print(f"\nSaved {MODEL_PATH.name} ({size_mb:.1f} MB) and {METRICS_PATH.name}")


if __name__ == "__main__":
    main()
