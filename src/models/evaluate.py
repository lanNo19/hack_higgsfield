"""Evaluation metrics for the two-stage churn pipeline."""
from __future__ import annotations

import numpy as np
from datetime import datetime
from sklearn.metrics import accuracy_score, average_precision_score, f1_score

from src.churn.thresholds import best_f1_threshold


def evaluate(
    name: str,
    y_binary: np.ndarray,
    y_volInv: np.ndarray,
    oof_s1: np.ndarray,
    oof_s2: np.ndarray,
    thr_s2: float = 0.5,
) -> dict:
    """
    Compute Accuracy, PR-AUC, F1 (binary + 3-class) and optimal threshold from OOF predictions.

    3-class labels: 0=not_churned, 1=vol_churn, 2=invol_churn.

    Args:
        name:      Pipeline identifier (e.g. "A")
        y_binary:  Ground truth — 0=not_churned, 1=churned
        y_volInv:  Ground truth — 0=vol_churn, 1=invol_churn (only valid where y_binary==1)
        oof_s1:    OOF P(Churn) from Stage 1
        oof_s2:    OOF P(Invol|Churn) from Stage 2
        thr_s2:    Threshold for Stage 2 invol classification (default 0.5)
    """
    churn_mask = y_binary == 1

    # Stage 1 metrics
    pr_auc_s1  = average_precision_score(y_binary, oof_s1)
    thr, f1_s1 = best_f1_threshold(y_binary, oof_s1)
    preds_s1   = (oof_s1 >= thr).astype(int)
    acc_s1     = accuracy_score(y_binary, preds_s1)

    # Stage 2 metrics (on churned subset only)
    pr_auc_s2 = (
        average_precision_score(y_volInv[churn_mask], oof_s2[churn_mask])
        if churn_mask.sum() > 0 else float("nan")
    )

    # 3-class metrics: combine both stages into not_churned/vol_churn/invol_churn
    # y_3: 0=not_churned, 1=vol_churn, 2=invol_churn
    y_3 = np.where(y_binary == 0, 0, np.where(y_volInv == 1, 2, 1))
    preds_s2   = (oof_s2 >= thr_s2).astype(int)
    preds_3    = np.where(preds_s1 == 0, 0, np.where(preds_s2 == 1, 2, 1))
    acc_3      = accuracy_score(y_3, preds_3)
    macro_f1_3 = f1_score(y_3, preds_3, average="macro")

    return {
        "pipeline":          name,
        "accuracy_3class":   round(acc_3,      4),
        "accuracy_s1":       round(acc_s1,      4),
        "pr_auc_s1":         round(pr_auc_s1,   4),
        "pr_auc_s2":         round(pr_auc_s2,   4),
        "best_f1_s1":        round(f1_s1,        4),
        "macro_f1_3class":   round(macro_f1_3,  4),
        "best_threshold_s1": round(thr,           3),
        "timestamp":         datetime.now().isoformat(),
    }
