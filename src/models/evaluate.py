"""Evaluation metrics for the two-stage churn pipeline."""
from __future__ import annotations

import numpy as np
from datetime import datetime
from sklearn.metrics import average_precision_score, f1_score

from src.churn.thresholds import best_f1_threshold


def evaluate(
    name: str,
    y_binary: np.ndarray,
    y_volInv: np.ndarray,
    oof_s1: np.ndarray,
    oof_s2: np.ndarray,
) -> dict:
    """
    Compute PR-AUC, best-F1, macro-F1, and optimal threshold from OOF predictions.

    Args:
        name:      Pipeline identifier (e.g. "A")
        y_binary:  Ground truth — 0=not_churned, 1=churned
        y_volInv:  Ground truth — 0=vol_churn, 1=invol_churn (only valid where y_binary==1)
        oof_s1:    OOF P(Churn) from Stage 1
        oof_s2:    OOF P(Invol|Churn) from Stage 2
    """
    churn_mask = y_binary == 1
    pr_auc_s1  = average_precision_score(y_binary, oof_s1)
    thr, f1_s1 = best_f1_threshold(y_binary, oof_s1)

    pr_auc_s2 = (
        average_precision_score(y_volInv[churn_mask], oof_s2[churn_mask])
        if churn_mask.sum() > 0 else float("nan")
    )
    preds_s1  = (oof_s1 >= thr).astype(int)
    macro_f1  = f1_score(y_binary, preds_s1, average="macro")

    return {
        "pipeline":          name,
        "pr_auc_s1":         round(pr_auc_s1,  4),
        "pr_auc_s2":         round(pr_auc_s2,  4),
        "best_f1_s1":        round(f1_s1,       4),
        "macro_f1_s1":       round(macro_f1,    4),
        "best_threshold_s1": round(thr,          3),
        "timestamp":         datetime.now().isoformat(),
    }
