"""Pipeline 3: Weighted Average Ensemble.

Goal: Quick ensemble gain over best single model.
Method: Optimize per-model weights on holdout set using log-loss minimization.
Requires: Pipeline 2 OOF outputs (run P02 first).
Expected time: ~5 minutes.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import log_loss

from src.models.pipeline_utils import (
    ARTIFACTS, evaluate_proba, load_train_data,
    make_holdout, save_oof, save_result, load_oof,
)
from src.utils.logger import get_logger

log = get_logger(__name__)
NAME = "P03_weighted_ensemble"


def optimize_weights(probas: list[np.ndarray], y: np.ndarray) -> np.ndarray:
    """Find weights w (sum=1, w>=0) minimising log-loss on y."""
    n_models = len(probas)
    stack = np.stack(probas, axis=0)  # (M, N, 3)

    def objective(w):
        blended = np.einsum("m,mnc->nc", w, stack)
        return log_loss(y, blended)

    constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1}]
    bounds = [(0, 1)] * n_models
    w0 = np.ones(n_models) / n_models

    result = minimize(objective, w0, method="SLSQP", bounds=bounds, constraints=constraints)
    weights = np.clip(result.x, 0, 1)
    weights /= weights.sum()
    return weights


def run() -> dict:
    log.info("=" * 60)
    log.info("Pipeline 3: Weighted Average Ensemble")
    log.info("=" * 60)

    X, y = load_train_data()
    X_tv, X_hold, y_tv, y_hold = make_holdout(X, y)

    # Load OOF predictions from P02 (trainval split)
    oof_lgbm = load_oof("p02_lgbm")
    oof_xgb = load_oof("p02_xgb")
    oof_cat = load_oof("p02_catboost")

    log.info("Optimizing ensemble weights on trainval OOF...")
    weights = optimize_weights([oof_lgbm, oof_xgb, oof_cat], y_tv)
    log.info("Optimized weights — LGB: %.3f  XGB: %.3f  CAT: %.3f", *weights)

    oof_blend = (
        weights[0] * oof_lgbm +
        weights[1] * oof_xgb +
        weights[2] * oof_cat
    )
    cv_result = evaluate_proba(f"{NAME}_oof", y_tv, oof_blend)
    save_result(cv_result)
    save_oof("p03", oof_blend)

    # Holdout evaluation using per-model holdout probas
    hold_lgbm = np.load(ARTIFACTS / "oof_p02_lgbm_holdout.npy")
    hold_xgb = np.load(ARTIFACTS / "oof_p02_xgb_holdout.npy")
    hold_cat = np.load(ARTIFACTS / "oof_p02_catboost_holdout.npy")

    hold_blend = weights[0] * hold_lgbm + weights[1] * hold_xgb + weights[2] * hold_cat
    hold_result = evaluate_proba(f"{NAME}_holdout", y_hold, hold_blend,
                                 extra={"lgbm_w": round(float(weights[0]), 4),
                                        "xgb_w": round(float(weights[1]), 4),
                                        "cat_w": round(float(weights[2]), 4)})
    save_result(hold_result)
    np.save(ARTIFACTS / "oof_p03_holdout.npy", hold_blend)

    return hold_result


if __name__ == "__main__":
    run()
