"""Evaluation utilities for direct 3-class models (K / L / M) and fixed cascade (N)."""
from __future__ import annotations

import numpy as np
from datetime import datetime
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    f1_score,
)

from src.utils.logger import get_logger

log = get_logger(__name__)

CLASS_NAMES = ["not_churned", "vol_churn", "invol_churn"]


# ── Direct multiclass evaluation (Pipelines K / L / M) ────────────────────────

def evaluate_multiclass(
    name: str,
    y_3: np.ndarray,
    oof_proba: np.ndarray,
) -> dict:
    """Compute metrics for a direct 3-class model.

    Args:
        name:      Pipeline identifier (e.g. "K")
        y_3:       Ground-truth labels: 0=not_churned, 1=vol_churn, 2=invol_churn
        oof_proba: OOF probability matrix, shape (n_samples, 3)

    Returns:
        Metrics dict compatible with the multiclass results CSV.
    """
    preds = np.argmax(oof_proba, axis=1)

    acc_3      = accuracy_score(y_3, preds)
    macro_f1   = f1_score(y_3, preds, average="macro",    zero_division=0)
    weighted_f1 = f1_score(y_3, preds, average="weighted", zero_division=0)
    f1_per     = f1_score(y_3, preds, average=None,       zero_division=0)

    # OvR PR-AUC per class and macro average
    y_onehot = np.eye(3)[y_3]
    pr_auc_macro = average_precision_score(y_onehot, oof_proba, average="macro")
    pr_auc_per   = [
        average_precision_score((y_3 == c).astype(int), oof_proba[:, c])
        for c in range(3)
    ]

    log.info(
        "Pipeline %s — acc: %.4f  macro_f1: %.4f  pr_auc_macro: %.4f",
        name, acc_3, macro_f1, pr_auc_macro,
    )
    log.info(
        "\n%s",
        classification_report(y_3, preds, target_names=CLASS_NAMES, zero_division=0),
    )

    return {
        "pipeline":            name,
        "accuracy_3class":     round(acc_3,            4),
        "macro_f1_3class":     round(macro_f1,         4),
        "weighted_f1_3class":  round(weighted_f1,      4),
        "pr_auc_macro":        round(pr_auc_macro,     4),
        "pr_auc_not_churned":  round(pr_auc_per[0],   4),
        "pr_auc_vol_churn":    round(pr_auc_per[1],   4),
        "pr_auc_invol_churn":  round(pr_auc_per[2],   4),
        "f1_not_churned":      round(float(f1_per[0]), 4),
        "f1_vol_churn":        round(float(f1_per[1]), 4),
        "f1_invol_churn":      round(float(f1_per[2]), 4),
        "timestamp":           datetime.now().isoformat(),
    }


# ── Joint threshold search (Pipeline N — fixed cascade) ───────────────────────

def joint_threshold_search(
    y_binary: np.ndarray,
    y_volInv: np.ndarray,
    oof_s1: np.ndarray,
    oof_s2: np.ndarray,
    grid_steps: int = 40,
) -> tuple[float, float, float]:
    """Grid search over (thr_s1, thr_s2) maximising 3-class macro F1.

    Args:
        y_binary:   0=not_churned, 1=churned
        y_volInv:   0=vol_churn,   1=invol_churn  (valid for all users;
                    only meaningful where y_binary==1)
        oof_s1:     OOF P(Churn)
        oof_s2:     OOF P(Invol|Churn)
        grid_steps: Number of threshold values to try per axis

    Returns:
        (best_thr_s1, best_thr_s2, best_macro_f1)
    """
    y_3 = np.where(y_binary == 0, 0, np.where(y_volInv == 1, 2, 1))
    grid = np.linspace(0.10, 0.90, grid_steps)

    best_f1   = -1.0
    best_thr1 = 0.5
    best_thr2 = 0.5

    for thr1 in grid:
        preds_s1 = (oof_s1 >= thr1).astype(int)
        for thr2 in grid:
            preds_s2  = (oof_s2 >= thr2).astype(int)
            preds_3   = np.where(preds_s1 == 0, 0, np.where(preds_s2 == 1, 2, 1))
            f1        = f1_score(y_3, preds_3, average="macro", zero_division=0)
            if f1 > best_f1:
                best_f1   = f1
                best_thr1 = float(thr1)
                best_thr2 = float(thr2)

    log.info(
        "Joint threshold search → thr_s1=%.3f  thr_s2=%.3f  macro_f1=%.4f",
        best_thr1, best_thr2, best_f1,
    )
    return best_thr1, best_thr2, best_f1


def evaluate_fixed_cascade(
    name: str,
    y_binary: np.ndarray,
    y_volInv: np.ndarray,
    oof_s1: np.ndarray,
    oof_s2: np.ndarray,
) -> dict:
    """Evaluate a cascade pipeline with BOTH binary-optimal and jointly-optimal thresholds.

    The side-by-side comparison makes the threshold-choice impact explicit.
    """
    from sklearn.metrics import average_precision_score
    from src.churn.thresholds import best_f1_threshold

    y_3 = np.where(y_binary == 0, 0, np.where(y_volInv == 1, 2, 1))

    # ── Cascade metrics with binary-optimal S1 threshold (like A–F) ──────────
    pr_auc_s1  = average_precision_score(y_binary, oof_s1)
    churn_mask = y_binary == 1
    pr_auc_s2  = (
        average_precision_score(y_volInv[churn_mask], oof_s2[churn_mask])
        if churn_mask.sum() > 0 else float("nan")
    )

    bin_thr, best_f1_bin = best_f1_threshold(y_binary, oof_s1)
    preds_bin  = np.where(
        (oof_s1 >= bin_thr), np.where(oof_s2 >= 0.5, 2, 1), 0
    )
    macro_f1_bin = f1_score(y_3, preds_bin, average="macro", zero_division=0)
    acc_bin      = accuracy_score(y_3, preds_bin)

    # ── Cascade metrics with jointly optimised thresholds ─────────────────────
    jt1, jt2, macro_f1_joint = joint_threshold_search(y_binary, y_volInv, oof_s1, oof_s2)
    preds_joint  = np.where(
        (oof_s1 >= jt1), np.where(oof_s2 >= jt2, 2, 1), 0
    )
    acc_joint    = accuracy_score(y_3, preds_joint)
    f1_per_joint = f1_score(y_3, preds_joint, average=None, zero_division=0)

    log.info(
        "Pipeline %s — pr_auc_s1: %.4f  pr_auc_s2: %.4f\n"
        "  binary thr (%.3f / 0.5) → macro_f1=%.4f  acc=%.4f\n"
        "  joint  thr (%.3f / %.3f) → macro_f1=%.4f  acc=%.4f",
        name,
        pr_auc_s1, pr_auc_s2,
        bin_thr, macro_f1_bin, acc_bin,
        jt1, jt2, macro_f1_joint, acc_joint,
    )
    log.info(
        "\n%s",
        classification_report(y_3, preds_joint, target_names=CLASS_NAMES, zero_division=0),
    )

    return {
        "pipeline":                  name,
        "pr_auc_s1":                 round(pr_auc_s1,     4),
        "pr_auc_s2":                 round(pr_auc_s2,     4),
        "best_f1_s1":                round(best_f1_bin,   4),
        "binary_thr_s1":             round(bin_thr,       3),
        # With binary-optimal threshold (equivalent to A–F for comparison)
        "macro_f1_binary_thr":       round(macro_f1_bin,  4),
        "accuracy_binary_thr":       round(acc_bin,       4),
        # With jointly-optimised thresholds
        "macro_f1_joint_thr":        round(macro_f1_joint, 4),
        "accuracy_joint_thr":        round(acc_joint,     4),
        "joint_thr_s1":              round(jt1,           3),
        "joint_thr_s2":              round(jt2,           3),
        "f1_not_churned_joint":      round(float(f1_per_joint[0]), 4),
        "f1_vol_churn_joint":        round(float(f1_per_joint[1]), 4),
        "f1_invol_churn_joint":      round(float(f1_per_joint[2]), 4),
        "timestamp":                 datetime.now().isoformat(),
    }
