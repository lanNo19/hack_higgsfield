"""Pipeline 6: Hierarchical + Flat Hybrid Ensemble.

Combine Pipeline 4 (stacking, flat) and Pipeline 5 (hierarchical two-stage)
by optimizing a blend weight on the holdout set.

Requires: Pipeline 4 and Pipeline 5 outputs.
Expected time: ~5 minutes.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.metrics import log_loss

from src.models.pipeline_utils import (
    ARTIFACTS, evaluate_proba, load_train_data,
    make_holdout, save_oof, save_result, load_oof,
)
from src.utils.logger import get_logger

log = get_logger(__name__)
NAME = "P06_hybrid"


def run() -> dict:
    log.info("=" * 60)
    log.info("Pipeline 6: Hierarchical + Flat Hybrid")
    log.info("=" * 60)

    X, y = load_train_data()
    X_tv, X_hold, y_tv, y_hold = make_holdout(X, y)

    # OOF predictions from P04 (stacking) and P05 (hierarchical) on trainval
    oof_flat = load_oof("p04")
    oof_hier = load_oof("p05")

    # Optimize blend weight alpha on trainval OOF
    def logloss(alpha):
        blended = alpha * oof_flat + (1 - alpha) * oof_hier
        return log_loss(y_tv, blended)

    result = minimize_scalar(logloss, bounds=(0, 1), method="bounded")
    alpha = float(result.x)
    log.info("Optimal alpha (flat weight): %.4f  (hier weight: %.4f)", alpha, 1 - alpha)

    oof_hybrid = alpha * oof_flat + (1 - alpha) * oof_hier
    cv_result = evaluate_proba(f"{NAME}_oof", y_tv, oof_hybrid,
                               extra={"alpha_flat": round(alpha, 4),
                                      "alpha_hier": round(1 - alpha, 4)})
    save_result(cv_result)
    save_oof("p06", oof_hybrid)

    # Holdout evaluation
    hold_flat = np.load(ARTIFACTS / "oof_p04_holdout.npy")
    hold_hier = np.load(ARTIFACTS / "oof_p05_holdout.npy")
    hold_hybrid = alpha * hold_flat + (1 - alpha) * hold_hier

    hold_result = evaluate_proba(f"{NAME}_holdout", y_hold, hold_hybrid,
                                 extra={"alpha_flat": round(alpha, 4),
                                        "alpha_hier": round(1 - alpha, 4)})
    save_result(hold_result)
    np.save(ARTIFACTS / "oof_p06_holdout.npy", hold_hybrid)

    return hold_result


if __name__ == "__main__":
    run()
