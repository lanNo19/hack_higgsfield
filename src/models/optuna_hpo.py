"""Optuna hyperparameter search for LightGBM Stage 1 (Pipeline G)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import lightgbm as lgb
import optuna
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold

from src.models.train import S1_FEATURES, safe_features
from src.utils.helpers import load_config
from src.utils.logger import get_logger

log = get_logger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)


def run_lgbm_study(
    X: pd.DataFrame,
    y_binary: np.ndarray,
) -> dict:
    """Run Optuna TPE search for LightGBM S1 hyper-params.

    Uses a fast 3-fold (configurable) CV with PR-AUC as objective.
    Returns the best params dict — caller passes these to build_lgbm_focal().
    """
    cfg = load_config().get("optuna", {})
    n_trials: int = cfg.get("n_trials", 80)
    timeout: int = cfg.get("timeout_seconds", 1800)
    cv_folds: int = cfg.get("cv_folds", 3)

    feat = safe_features(X, S1_FEATURES)
    Xf = X[feat]

    def objective(trial: optuna.Trial) -> float:
        params = dict(
            n_estimators=trial.suggest_int("n_estimators", 300, 1500),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            num_leaves=trial.suggest_int("num_leaves", 31, 255),
            min_child_samples=trial.suggest_int("min_child_samples", 10, 100),
            subsample=trial.suggest_float("subsample", 0.5, 1.0),
            subsample_freq=1,
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            is_unbalance=True,
            n_jobs=-1,
            random_state=42,
            verbose=-1,
        )
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        scores = []
        for train_idx, val_idx in cv.split(Xf, y_binary):
            m = lgb.LGBMClassifier(**params)
            m.fit(
                Xf.iloc[train_idx], y_binary[train_idx],
                eval_set=[(Xf.iloc[val_idx], y_binary[val_idx])],
                callbacks=[
                    lgb.early_stopping(stopping_rounds=50, verbose=False),
                    lgb.log_evaluation(-1),
                ],
            )
            scores.append(
                average_precision_score(y_binary[val_idx],
                                        m.predict_proba(Xf.iloc[val_idx])[:, 1])
            )
        return float(np.mean(scores))

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
    )
    log.info("Starting Optuna study: %d trials, timeout=%ds, cv=%d-fold",
             n_trials, timeout, cv_folds)
    study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=False)

    log.info("Optuna best PR-AUC (%d-fold): %.4f", cv_folds, study.best_value)
    log.info("Best params: %s", study.best_params)
    return study.best_params
