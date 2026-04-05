"""Pipeline 5: Hierarchical Two-Stage Decomposition.

Stage 1: Binary classifier — churn vs not_churn (perfectly balanced 50/50).
Stage 2: Binary classifier — vol_churn vs invol_churn (trained only on churn samples).

Soft handoff: multiply stage probabilities instead of hard thresholding.
    P(not_churned)  = P_s1(not_churn)
    P(vol_churn)    = P_s1(churn) × P_s2(vol)
    P(invol_churn)  = P_s1(churn) × P_s2(invol)

Expected time: ~1 hour.
"""
from __future__ import annotations

import numpy as np
import lightgbm as lgb
import optuna
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

from src.models.pipeline_utils import (
    ARTIFACTS, evaluate_proba, hierarchical_to_3class,
    load_train_data, make_holdout, save_oof, save_result, LGBM_DEVICE,
)
from src.utils.logger import get_logger

log = get_logger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)
NAME = "P05_hierarchical"
N_TRIALS = 40
CV_FOLDS = 5


def _tune_binary_lgbm(X, y_bin, label: str, n_trials: int = N_TRIALS) -> dict:
    """Tune LightGBM for a binary task with Optuna."""
    log.info("Tuning %s LightGBM (%d trials)...", label, n_trials)

    def objective(trial: optuna.Trial) -> float:
        params = dict(
            n_estimators=trial.suggest_int("n_estimators", 200, 1500),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            num_leaves=trial.suggest_int("num_leaves", 15, 255),
            min_child_samples=trial.suggest_int("min_child_samples", 5, 100),
            subsample=trial.suggest_float("subsample", 0.5, 1.0),
            subsample_freq=1,
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.3, 1.0),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            class_weight="balanced",
            n_jobs=-1, random_state=42, verbose=-1, device=LGBM_DEVICE,
        )
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        scores = []
        for tr, val in cv.split(X, y_bin):
            m = lgb.LGBMClassifier(**params)
            m.fit(X.iloc[tr], y_bin[tr])
            scores.append(f1_score(y_bin[val], m.predict(X.iloc[val]), average="macro", zero_division=0))
        return float(np.mean(scores))

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    log.info("%s best f1=%.4f", label, study.best_value)
    return study.best_params


def run(n_trials: int = N_TRIALS) -> dict:
    log.info("=" * 60)
    log.info("Pipeline 5: Hierarchical Two-Stage")
    log.info("=" * 60)

    X, y = load_train_data()
    X_tv, X_hold, y_tv, y_hold = make_holdout(X, y)

    # Stage labels
    y_s1_tv = (y_tv > 0).astype(int)           # 0=not_churn, 1=churn
    y_s1_hold = (y_hold > 0).astype(int)

    # Stage 2 labels: 1=invol_churn (label 2), 0=vol_churn (label 1)
    y_s2_tv = (y_tv == 2).astype(int)
    y_s2_hold = (y_hold == 2).astype(int)

    # ── Tune stage models ──────────────────────────────────────────────────────
    params_s1 = _tune_binary_lgbm(X_tv, y_s1_tv, "Stage1(churn)", n_trials)

    # Stage 2 only trained on churn samples
    churn_mask_tv = y_s1_tv == 1
    X_churn = X_tv[churn_mask_tv]
    y_churn_s2 = y_s2_tv[churn_mask_tv]
    params_s2 = _tune_binary_lgbm(X_churn, y_churn_s2, "Stage2(vol/invol)", n_trials)

    # ── OOF generation ────────────────────────────────────────────────────────
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=42)
    oof_s1 = np.zeros(len(X_tv))    # P(churn)
    oof_s2 = np.zeros(len(X_tv))    # P(invol|churn)

    for fold, (tr, val) in enumerate(cv.split(X_tv, y_s1_tv)):
        # Stage 1
        m1 = lgb.LGBMClassifier(n_jobs=-1, random_state=42, verbose=-1, device=LGBM_DEVICE,
                                  class_weight="balanced", **params_s1)
        m1.fit(X_tv.iloc[tr], y_s1_tv[tr])
        oof_s1[val] = m1.predict_proba(X_tv.iloc[val])[:, 1]

        # Stage 2: train only on churn subset of train fold
        churn_tr = tr[y_s1_tv[tr] == 1]
        churn_val = val[y_s1_tv[val] == 1]
        if len(churn_tr) > 20 and len(churn_val) > 0:
            m2 = lgb.LGBMClassifier(n_jobs=-1, random_state=42, verbose=-1, device=LGBM_DEVICE,
                                     class_weight="balanced", **params_s2)
            m2.fit(X_tv.iloc[churn_tr], y_s2_tv[churn_tr])
            oof_s2[churn_val] = m2.predict_proba(X_tv.iloc[churn_val])[:, 1]

        log.info("Fold %d done", fold + 1)

    oof_3class = hierarchical_to_3class(oof_s1, oof_s2)
    cv_result = evaluate_proba(f"{NAME}_oof", y_tv, oof_3class)
    save_result(cv_result)
    save_oof("p05", oof_3class)

    # ── Holdout evaluation ────────────────────────────────────────────────────
    # Retrain on full trainval
    m1_final = lgb.LGBMClassifier(n_jobs=-1, random_state=42, verbose=-1, device=LGBM_DEVICE,
                                    class_weight="balanced", **params_s1)
    m1_final.fit(X_tv, y_s1_tv)
    hold_s1 = m1_final.predict_proba(X_hold)[:, 1]

    m2_final = lgb.LGBMClassifier(n_jobs=-1, random_state=42, verbose=-1, device=LGBM_DEVICE,
                                    class_weight="balanced", **params_s2)
    m2_final.fit(X_churn, y_churn_s2)
    hold_s2 = m2_final.predict_proba(X_hold)[:, 1]

    hold_3class = hierarchical_to_3class(hold_s1, hold_s2)
    hold_result = evaluate_proba(f"{NAME}_holdout", y_hold, hold_3class)
    save_result(hold_result)
    np.save(ARTIFACTS / "oof_p05_holdout.npy", hold_3class)

    return hold_result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-trials", type=int, default=N_TRIALS)
    args = parser.parse_args()
    run(n_trials=args.n_trials)
