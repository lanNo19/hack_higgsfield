"""Threshold selection utilities for Stage 1 and Stage 2 classifiers."""
import numpy as np
from sklearn.metrics import precision_recall_curve


def best_f1_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[float, float]:
    """Return (threshold, f1) that maximises F1 on the given OOF predictions."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)
    idx = np.argmax(f1[:-1])   # thresholds has one fewer element than f1
    return float(thresholds[idx]), float(f1[idx])
