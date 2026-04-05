"""Pipeline 1: Single LightGBM Baseline (Flat Multiclass).

Goal: Establish baseline performance fast.
Model: LightGBM, objective='multiclass', class_weight='balanced'
Tuning: Manual defaults — no Optuna.
Expected time: ~5 minutes.
"""
from __future__ import annotations

import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold

from src.models.pipeline_utils import (
    CLASS_NAMES, evaluate_proba, load_train_data,
    make_holdout, save_oof, save_result,
)
from src.utils.logger import get_logger

log = get_logger(__name__)
NAME = "P01_lgbm_baseline"


def run() -> dict:
    log.info("=" * 60)
    log.info("Pipeline 1: LightGBM Baseline")
    log.info("=" * 60)

    X, y = load_train_data()
    X_tv, X_hold, y_tv, y_hold = make_holdout(X, y)

    params = dict(
        objective="multiclass",
        num_class=3,
        n_estimators=1000,
        learning_rate=0.05,
        num_leaves=63,
        min_child_samples=30,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42,
        verbose=-1,
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_tv = np.zeros((len(X_tv), 3))

    for fold, (tr_idx, val_idx) in enumerate(cv.split(X_tv, y_tv)):
        X_tr, X_val = X_tv.iloc[tr_idx], X_tv.iloc[val_idx]
        y_tr, y_val = y_tv[tr_idx], y_tv[val_idx]

        model = lgb.LGBMClassifier(**params)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            callbacks=[
                lgb.early_stopping(stopping_rounds=50, verbose=False),
                lgb.log_evaluation(period=200),
            ],
        )
        oof_tv[val_idx] = model.predict_proba(X_val)
        log.info("Fold %d done", fold + 1)

    # Evaluate on CV (trainval OOF) and holdout
    cv_result = evaluate_proba(f"{NAME}_oof", y_tv, oof_tv)

    # Retrain on full trainval for holdout evaluation
    final_model = lgb.LGBMClassifier(**params)
    final_model.fit(X_tv, y_tv)
    hold_proba = final_model.predict_proba(X_hold)
    hold_result = evaluate_proba(f"{NAME}_holdout", y_hold, hold_proba)

    # Save OOF on full trainval for ensembling (pipelines 3, 6)
    save_oof("p01", oof_tv)

    # Persist holdout labels/proba for downstream pipelines
    from src.models.pipeline_utils import ARTIFACTS
    np.save(ARTIFACTS / "y_holdout.npy", y_hold)
    np.save(ARTIFACTS / "oof_p01_holdout.npy", hold_proba)

    save_result(cv_result)
    save_result(hold_result)
    return hold_result


if __name__ == "__main__":
    run()
