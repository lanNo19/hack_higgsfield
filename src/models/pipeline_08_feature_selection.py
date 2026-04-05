"""Pipeline 8: Feature Selection Ablation.

Tests multiple feature selection approaches on best single model (LightGBM).
Methods:
  1. all_features — baseline with all features
  2. top80_permutation — top 80 by permutation importance
  3. top50_permutation — top 50 by permutation importance
  4. vif_filtered — remove features with VIF > 10 (multicollinearity)
  5. correlation_filtered — remove one of any pair with |r| > 0.95

Evaluation: Macro F1 on 5-fold CV.
Expected time: ~45 minutes.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.inspection import permutation_importance
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

from src.models.pipeline_utils import (
    evaluate_proba, load_train_data, make_holdout, save_result, LGBM_DEVICE,
)
from src.utils.logger import get_logger

log = get_logger(__name__)


# ── Feature selection utilities ────────────────────────────────────────────────

def _get_permutation_top_k(X: pd.DataFrame, y: np.ndarray, k: int) -> list[str]:
    """Train a quick LightGBM, compute permutation importance, return top-k features."""
    log.info("Computing permutation importance (top %d)...", k)
    m = lgb.LGBMClassifier(
        objective="multiclass", num_class=3, n_estimators=200,
        learning_rate=0.1, class_weight="balanced",
        n_jobs=-1, random_state=42, verbose=-1, device=LGBM_DEVICE,
    )
    m.fit(X, y)
    perm = permutation_importance(m, X, y, n_repeats=5, random_state=42,
                                  scoring="f1_macro", n_jobs=-1)
    indices = np.argsort(perm.importances_mean)[::-1][:k]
    selected = [X.columns[i] for i in indices]
    log.info("  Selected %d features", len(selected))
    return selected


def _remove_high_vif(X: pd.DataFrame, threshold: float = 10.0) -> list[str]:
    """Iteratively remove the feature with highest VIF until all VIF < threshold."""
    from statsmodels.stats.outliers_influence import variance_inflation_factor  # type: ignore[import]
    log.info("Computing VIF filtering (threshold=%.1f)...", threshold)
    cols = list(X.columns)
    while True:
        vif_data = pd.DataFrame({
            "feature": cols,
            "vif": [variance_inflation_factor(X[cols].values, i) for i in range(len(cols))],
        })
        max_vif = vif_data["vif"].max()
        if max_vif <= threshold:
            break
        worst = vif_data.loc[vif_data["vif"].idxmax(), "feature"]
        log.info("  Removing %s (VIF=%.1f)", worst, max_vif)
        cols.remove(worst)
    log.info("  %d features remaining after VIF filtering", len(cols))
    return cols


def _remove_high_correlation(X: pd.DataFrame, threshold: float = 0.95) -> list[str]:
    """Remove one of any feature pair with |r| > threshold."""
    log.info("Correlation filtering (threshold=%.2f)...", threshold)
    corr = X.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > threshold)]
    remaining = [c for c in X.columns if c not in to_drop]
    log.info("  Dropped %d, kept %d features", len(to_drop), len(remaining))
    return remaining


def _cv_oof_features(X: pd.DataFrame, y: np.ndarray, features: list[str],
                     n_splits: int = 5) -> tuple[np.ndarray, float]:
    """Run 5-fold CV on selected features, return OOF + macro F1."""
    Xf = X[features]
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    oof = np.zeros((len(Xf), 3))

    for tr, val in cv.split(Xf, y):
        m = lgb.LGBMClassifier(
            objective="multiclass", num_class=3, n_estimators=500,
            learning_rate=0.05, num_leaves=63, class_weight="balanced",
            n_jobs=-1, random_state=42, verbose=-1, device=LGBM_DEVICE,
        )
        m.fit(Xf.iloc[tr], y[tr])
        oof[val] = m.predict_proba(Xf.iloc[val])

    macro_f1 = f1_score(y, np.argmax(oof, axis=1), average="macro", zero_division=0)
    return oof, macro_f1


def run() -> list[dict]:
    log.info("=" * 60)
    log.info("Pipeline 8: Feature Selection Ablation")
    log.info("=" * 60)

    X, y = load_train_data()
    X_tv, X_hold, y_tv, y_hold = make_holdout(X, y)

    all_features = list(X_tv.columns)
    log.info("Total features: %d", len(all_features))

    # Precompute feature subsets
    feature_sets = {"all_features": all_features}

    try:
        feature_sets["top80_permutation"] = _get_permutation_top_k(X_tv, y_tv, 80)
        feature_sets["top50_permutation"] = _get_permutation_top_k(X_tv, y_tv, 50)
    except Exception as e:
        log.warning("Permutation importance failed: %s", e)

    try:
        feature_sets["correlation_filtered"] = _remove_high_correlation(X_tv)
    except Exception as e:
        log.warning("Correlation filter failed: %s", e)

    try:
        feature_sets["vif_filtered"] = _remove_high_vif(X_tv)
    except Exception as e:
        log.warning("VIF filter failed (statsmodels not installed?): %s", e)

    results = []
    for set_name, features in feature_sets.items():
        log.info("Running: %s (%d features)", set_name, len(features))
        try:
            oof, macro_f1 = _cv_oof_features(X_tv, y_tv, features)
            result = evaluate_proba(f"P08_{set_name}_oof", y_tv, oof,
                                    extra={"n_features": len(features), "feature_set": set_name})
            save_result(result)
            results.append(result)
        except Exception as e:
            log.warning("  %s failed: %s", set_name, e)

    log.info("\n── Feature Selection Results ──")
    for r in results:
        log.info("  %-40s  n_feat=%-4s  macro_f1=%s",
                 r.get("pipeline", "?"), r.get("n_features", "?"), r.get("macro_f1", "ERR"))

    return results


if __name__ == "__main__":
    run()
