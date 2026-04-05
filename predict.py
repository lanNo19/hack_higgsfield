"""Generate predictions on the test set using the trained CatBoost model.

Loads: models/trained/catboost_final.cbm
Input: data/processed/features_test.parquet
Output: predictions/predictions.csv  (user_id, churn_status, p_not_churned, p_vol_churn, p_invol_churn)

Usage:
    uv run python predict.py
    uv run python predict.py --model models/trained/catboost_final.cbm --out predictions/predictions.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

from src.utils.helpers import processed_path, root_path
from src.utils.logger import get_logger

log = get_logger(__name__)

CLASS_NAMES = ["not_churned", "vol_churn", "invol_churn"]
DEFAULT_MODEL = root_path() / "models" / "trained" / "catboost_final.cbm"
DEFAULT_OUT = root_path() / "predictions" / "predictions.csv"


def load_test_features() -> tuple[pd.DataFrame, pd.Series | None]:
    path = processed_path() / "features_test.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run: uv run python -m src.features.feature_engineering --mode test"
        )
    df = pd.read_parquet(path)
    log.info("Loaded test features: %d samples, %d columns", df.shape[0], df.shape[1])

    user_ids = df["user_id"] if "user_id" in df.columns else None
    X = df.drop(columns=["user_id", "churn_status"], errors="ignore")
    return X, user_ids


def main(model_path: Path = DEFAULT_MODEL, out_path: Path = DEFAULT_OUT) -> None:
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}. Run: uv run python train_final_model.py"
        )

    log.info("Loading model from %s", model_path)
    model = CatBoostClassifier()
    model.load_model(str(model_path))

    X, user_ids = load_test_features()

    # Align features to what the model was trained on
    train_features = model.feature_names_
    if train_features is not None:
        missing = [f for f in train_features if f not in X.columns]
        extra = [f for f in X.columns if f not in train_features]
        if missing:
            log.warning("Filling %d missing features with 0: %s", len(missing), missing)
            for f in missing:
                X[f] = 0
        if extra:
            log.warning("Dropping %d extra features not seen during training", len(extra))
        X = X[train_features]

    log.info("Running inference on %d samples...", len(X))
    proba = model.predict_proba(X)
    pred_class_idx = np.argmax(proba, axis=1)
    pred_labels = [CLASS_NAMES[i] for i in pred_class_idx]

    out = pd.DataFrame({
        "predicted_churn_status": pred_labels,
        "p_not_churned": proba[:, 0],
        "p_vol_churn":   proba[:, 1],
        "p_invol_churn": proba[:, 2],
    })
    if user_ids is not None:
        out.insert(0, "user_id", user_ids.values)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    log.info("Predictions saved → %s  (%d rows)", out_path, len(out))

    # Summary
    counts = out["predicted_churn_status"].value_counts()
    log.info("Prediction distribution:\n%s", counts.to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    main(model_path=args.model, out_path=args.out)
