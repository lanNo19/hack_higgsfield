"""Shared utilities for all strategy pipelines (Pipelines 1-12)."""
from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    f1_score,
    log_loss,
)
from sklearn.model_selection import train_test_split

from src.utils.helpers import processed_path, root_path
from src.utils.logger import get_logger

log = get_logger(__name__)

LABEL_MAP = {"not_churned": 0, "vol_churn": 1, "invol_churn": 2}
CLASS_NAMES = ["not_churned", "vol_churn", "invol_churn"]
ARTIFACTS = root_path() / "models" / "artifacts"
RESULTS_CSV = ARTIFACTS / "results_strategy.csv"


# ── Data loading ───────────────────────────────────────────────────────────────

def load_train_data() -> tuple[pd.DataFrame, np.ndarray]:
    """Load feature matrix + labels. Handles both old and new parquet formats.

    New format (feature_engineering.py): features_train.parquet contains
    'churn_status' as a string column alongside features.

    Old format (build_features.py): labels are in labels_train.parquet
    as integer-encoded values.
    """
    proc = processed_path()
    raw = pd.read_parquet(proc / "features_train.parquet")

    if "churn_status" in raw.columns:
        y_raw = raw["churn_status"]
        raw = raw.drop(columns=["churn_status"])
    else:
        y_raw = pd.read_parquet(proc / "labels_train.parquet").squeeze()

    drop = [c for c in ["user_id", "Unnamed: 0"] if c in raw.columns]
    X = raw.drop(columns=drop).select_dtypes(include=[np.number]).fillna(0)

    if y_raw.dtype == object:
        y = y_raw.map(LABEL_MAP).values.astype(int)
    else:
        y = y_raw.values.astype(int)

    log.info("Loaded %d samples, %d features. Labels: %s",
             len(X), X.shape[1], dict(zip(*np.unique(y, return_counts=True))))
    return X, y


def make_holdout(
    X: pd.DataFrame,
    y: np.ndarray,
    holdout_size: float = 0.15,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    """Split into 85% trainval + 15% holdout (stratified). Returns X_tv, X_hold, y_tv, y_hold."""
    return train_test_split(X, y, test_size=holdout_size, stratify=y, random_state=random_state)


# ── Evaluation ─────────────────────────────────────────────────────────────────

def evaluate_proba(
    name: str,
    y_true: np.ndarray,
    proba: np.ndarray,
    extra: dict | None = None,
) -> dict:
    """Compute all metrics for a 3-class probability output."""
    preds = np.argmax(proba, axis=1)
    y_oh = np.eye(3)[y_true]

    result = {
        "pipeline": name,
        "macro_f1": round(f1_score(y_true, preds, average="macro", zero_division=0), 4),
        "weighted_f1": round(f1_score(y_true, preds, average="weighted", zero_division=0), 4),
        "accuracy": round(accuracy_score(y_true, preds), 4),
        "pr_auc_macro": round(average_precision_score(y_oh, proba, average="macro"), 4),
        "logloss": round(log_loss(y_true, proba), 4),
        "f1_not_churned": round(f1_score(y_true, preds, labels=[0], average="macro", zero_division=0), 4),
        "f1_vol_churn": round(f1_score(y_true, preds, labels=[1], average="macro", zero_division=0), 4),
        "f1_invol_churn": round(f1_score(y_true, preds, labels=[2], average="macro", zero_division=0), 4),
        "timestamp": datetime.now().isoformat(),
    }
    if extra:
        result.update(extra)

    log.info("%-35s  macro_f1=%.4f  logloss=%.4f  pr_auc=%.4f",
             name, result["macro_f1"], result["logloss"], result["pr_auc_macro"])
    log.info("\n%s", classification_report(y_true, preds, target_names=CLASS_NAMES, zero_division=0))
    return result


# ── Hierarchical probability combination ───────────────────────────────────────

def hierarchical_to_3class(p_churn: np.ndarray, p_invol: np.ndarray) -> np.ndarray:
    """Combine stage 1 (P churn) and stage 2 (P invol|churn) into 3-class proba.

    Convention matches existing make_labels():
        p_invol: P(invol_churn | churned)  — 1=invol, 0=vol
    """
    proba = np.zeros((len(p_churn), 3))
    proba[:, 0] = 1 - p_churn               # not_churned
    proba[:, 2] = p_churn * p_invol          # invol_churn
    proba[:, 1] = p_churn * (1 - p_invol)   # vol_churn
    return proba


# ── Results persistence ────────────────────────────────────────────────────────

def save_result(result: dict) -> None:
    """Append a result dict to models/artifacts/results_strategy.csv."""
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    write_header = not RESULTS_CSV.exists()
    with open(RESULTS_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(result.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(result)
    log.info("Result saved → %s", RESULTS_CSV)


def save_oof(name: str, proba: np.ndarray) -> None:
    """Save OOF probability array to models/artifacts/oof_{name}.npy."""
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    path = ARTIFACTS / f"oof_{name}.npy"
    np.save(path, proba)
    log.info("OOF saved → %s  shape=%s", path, proba.shape)


def load_oof(name: str) -> np.ndarray:
    """Load OOF probability array saved by save_oof()."""
    path = ARTIFACTS / f"oof_{name}.npy"
    if not path.exists():
        raise FileNotFoundError(f"OOF file not found: {path}. Run the corresponding pipeline first.")
    return np.load(path)
