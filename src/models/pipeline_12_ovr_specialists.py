"""Pipeline 12: One-vs-Rest Specialist Classifiers.

Train 3 independent binary LightGBM classifiers:
  A: vol_churn vs rest
  B: invol_churn vs rest
  C: not_churned vs rest

Each tuned independently with Optuna (30 trials each).
Final prediction: argmax of calibrated (Platt-scaled) probabilities.

Expected time: ~1 hour.
"""
from __future__ import annotations

import numpy as np
import optuna
import lightgbm as lgb
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import f1_score, average_precision_score
from sklearn.model_selection import StratifiedKFold

from src.models.pipeline_utils import (
    ARTIFACTS, evaluate_proba, load_train_data,
    make_holdout, save_oof, save_result,
)
from src.utils.logger import get_logger

log = get_logger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)
NAME = "P12_ovr_specialists"
N_TRIALS = 30
CV_FOLDS = 5


def _tune_binary(X, y_bin, label: str, n_trials: int = N_TRIALS) -> dict:
    """Tune a binary LightGBM with Optuna, using PR-AUC as objective."""
    log.info("Tuning OvR specialist: %s (%d trials)...", label, n_trials)

    def objective(trial: optuna.Trial) -> float:
        params = dict(
            n_estimators=trial.suggest_int("n_estimators", 200, 1500),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            num_leaves=trial.suggest_int("num_leaves", 15, 127),
            min_child_samples=trial.suggest_int("min_child_samples", 5, 100),
            subsample=trial.suggest_float("subsample", 0.5, 1.0),
            subsample_freq=1,
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.3, 1.0),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            class_weight="balanced",
            n_jobs=-1, random_state=42, verbose=-1,
        )
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        scores = []
        for tr, val in cv.split(X, y_bin):
            m = lgb.LGBMClassifier(**params)
            m.fit(X.iloc[tr], y_bin[tr])
            proba = m.predict_proba(X.iloc[val])[:, 1]
            scores.append(average_precision_score(y_bin[val], proba))
        return float(np.mean(scores))

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    log.info("%s best PR-AUC=%.4f", label, study.best_value)
    return study.best_params


def _oof_binary(params, X, y_bin, n_splits=CV_FOLDS) -> np.ndarray:
    """Generate OOF P(class=1) with Platt scaling calibration."""
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    oof = np.zeros(len(X))

    for tr, val in cv.split(X, y_bin):
        base = lgb.LGBMClassifier(n_jobs=-1, random_state=42, verbose=-1,
                                   class_weight="balanced", **params)
        cal = CalibratedClassifierCV(base, cv=3, method="sigmoid")
        cal.fit(X.iloc[tr], y_bin[tr])
        oof[val] = cal.predict_proba(X.iloc[val])[:, 1]

    return oof


def run(n_trials: int = N_TRIALS) -> dict:
    log.info("=" * 60)
    log.info("Pipeline 12: OvR Specialist Classifiers")
    log.info("=" * 60)

    X, y = load_train_data()
    X_tv, X_hold, y_tv, y_hold = make_holdout(X, y)

    # Binary targets for each class
    targets = {
        "vol_churn":   (y_tv == 1).astype(int),
        "invol_churn": (y_tv == 2).astype(int),
        "not_churned": (y_tv == 0).astype(int),
    }
    targets_hold = {
        "vol_churn":   (y_hold == 1).astype(int),
        "invol_churn": (y_hold == 2).astype(int),
        "not_churned": (y_hold == 0).astype(int),
    }

    # ── Tune + OOF for each specialist ────────────────────────────────────────
    oof_per_class = {}
    hold_per_class = {}
    params_per_class = {}

    for cls_name, y_bin_tv in targets.items():
        params = _tune_binary(X_tv, y_bin_tv, cls_name, n_trials)
        params_per_class[cls_name] = params
        oof_per_class[cls_name] = _oof_binary(params, X_tv, y_bin_tv)

        # Holdout: retrain calibrated model on full trainval
        base = lgb.LGBMClassifier(n_jobs=-1, random_state=42, verbose=-1,
                                   class_weight="balanced", **params)
        cal = CalibratedClassifierCV(base, cv=5, method="sigmoid")
        cal.fit(X_tv, y_bin_tv)
        hold_per_class[cls_name] = cal.predict_proba(X_hold)[:, 1]

    # ── Combine 3 binary probabilities → 3-class matrix ───────────────────────
    # Stack as [P(not_churned), P(vol_churn), P(invol_churn)] and normalize
    def _stack_and_normalize(per_class: dict) -> np.ndarray:
        proba = np.column_stack([
            per_class["not_churned"],
            per_class["vol_churn"],
            per_class["invol_churn"],
        ])
        row_sums = proba.sum(axis=1, keepdims=True).clip(min=1e-9)
        return proba / row_sums

    oof_3class = _stack_and_normalize(oof_per_class)
    hold_3class = _stack_and_normalize(hold_per_class)

    cv_result = evaluate_proba(f"{NAME}_oof", y_tv, oof_3class)
    save_result(cv_result)
    save_oof("p12", oof_3class)

    hold_result = evaluate_proba(f"{NAME}_holdout", y_hold, hold_3class)
    save_result(hold_result)
    np.save(ARTIFACTS / "oof_p12_holdout.npy", hold_3class)

    return hold_result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-trials", type=int, default=N_TRIALS)
    args = parser.parse_args()
    run(n_trials=args.n_trials)
