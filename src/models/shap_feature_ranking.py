"""
Compute SHAP-based feature importance for the 3-class churn task.

Trains a quick LightGBM multiclass on all S1_FEATURES (single 80/20 split —
no full CV needed just for importance ranking), computes SHAP values via
TreeExplainer, aggregates mean |SHAP| across all 3 classes, and saves:

  models/artifacts/shap_feature_ranking.csv   — ranked feature table
  models/artifacts/feature_lists.json          — top-N lists: 25/50/75/100/all

Run:
    uv run python -m src.models.shap_feature_ranking
"""
from __future__ import annotations

import json

import lightgbm as lgb
import numpy as np
import pandas as pd
import shap
from sklearn.model_selection import train_test_split

from src.models.train import S1_FEATURES, load_feature_matrix, make_splits, safe_features
from src.utils.helpers import root_path
from src.utils.logger import get_logger

log = get_logger(__name__)
_ARTIFACTS = root_path() / "models" / "artifacts"


def compute_shap_ranking(n_estimators: int = 500) -> pd.DataFrame:
    """Fit a quick LGBM on 80% of train, compute SHAP on the held-out 20%.

    Returns a DataFrame ranked by mean |SHAP| across all 3 classes.
    Side-effects: writes shap_feature_ranking.csv and feature_lists.json.
    """
    X, y = load_feature_matrix()
    X_train, _, _, y_train, _, _ = make_splits(X, y)

    feat = safe_features(X_train, S1_FEATURES)
    y_3  = y_train.values

    log.info("Feature matrix: %d samples × %d features", len(X_train), len(feat))

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train[feat], y_3, test_size=0.20, stratify=y_3, random_state=42
    )

    # Quick fit — intentionally simple, we only need stable importances
    model = lgb.LGBMClassifier(
        objective="multiclass",
        num_class=3,
        n_estimators=n_estimators,
        learning_rate=0.05,
        num_leaves=63,
        min_child_samples=30,
        subsample=0.80,
        colsample_bytree=0.70,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42,
        verbose=-1,
    )
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        callbacks=[
            lgb.early_stopping(100, verbose=False),
            lgb.log_evaluation(100),
        ],
    )
    log.info("Model fitted. Computing SHAP values on %d samples...", len(X_val))

    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_val)

    # shap_values can be:
    #   - list of n_classes arrays (n_val, n_feat)   — older SHAP
    #   - ndarray (n_val, n_feat, n_classes)          — newer SHAP
    if isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        mean_abs = np.abs(shap_values).mean(axis=(0, 2))   # mean over samples & classes
    else:
        mean_abs = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)

    ranking = (
        pd.DataFrame({"feature": feat, "mean_abs_shap": mean_abs})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )
    ranking.index += 1
    ranking.index.name = "rank"

    _ARTIFACTS.mkdir(parents=True, exist_ok=True)
    ranking.to_csv(_ARTIFACTS / "shap_feature_ranking.csv")
    log.info("Saved shap_feature_ranking.csv (%d features ranked)", len(ranking))

    # Persist top-N lists
    ordered = ranking["feature"].tolist()
    feature_lists: dict[str, list[str]] = {
        "all":    ordered,
        "top100": ordered[:100],
        "top75":  ordered[:75],
        "top50":  ordered[:50],
        "top25":  ordered[:25],
    }
    with open(_ARTIFACTS / "feature_lists.json", "w") as fh:
        json.dump(feature_lists, fh, indent=2)
    log.info("Saved feature_lists.json with keys: %s", list(feature_lists.keys()))

    return ranking


def load_feature_lists() -> dict[str, list[str]]:
    """Load the pre-computed top-N feature lists.  Run compute_shap_ranking() first."""
    path = _ARTIFACTS / "feature_lists.json"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `uv run python -m src.models.shap_feature_ranking` first."
        )
    with open(path) as fh:
        return json.load(fh)


if __name__ == "__main__":
    ranking = compute_shap_ranking()
    print("\nTop 30 features by mean |SHAP|:")
    print(ranking.head(30).to_string())
