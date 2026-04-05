"""Pipeline 2: Optuna-Tuned Big Three (LightGBM + XGBoost + CatBoost).

Goal: Best individual model performance from each booster.
Tuning: Optuna TPE sampler, MedianPruner, n_trials=50 each.
Outputs: Best params + OOF predictions saved for Pipelines 3 & 4.
Expected time: 1-2 hours.
"""
from __future__ import annotations

import json
import numpy as np
import optuna
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

from src.models.pipeline_utils import (
    ARTIFACTS, evaluate_proba, load_train_data,
    make_holdout, save_oof, save_result,
)
from src.utils.logger import get_logger

log = get_logger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)

N_TRIALS = 50
CV_FOLDS = 5


def _cv_macro_f1(model_fn, X, y, n_splits=3) -> float:
    """Fast 3-fold CV macro F1 for Optuna objectives."""
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = []
    for tr, val in cv.split(X, y):
        m = model_fn()
        m.fit(X.iloc[tr], y[tr])
        preds = np.argmax(m.predict_proba(X.iloc[val]), axis=1)
        scores.append(f1_score(y[val], preds, average="macro", zero_division=0))
    return float(np.mean(scores))


# ── LightGBM ──────────────────────────────────────────────────────────────────

def _lgbm_objective(trial: optuna.Trial, X: np.ndarray, y: np.ndarray) -> float:
    params = dict(
        objective="multiclass",
        num_class=3,
        n_estimators=trial.suggest_int("n_estimators", 300, 2000),
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        num_leaves=trial.suggest_int("num_leaves", 15, 255),
        min_child_samples=trial.suggest_int("min_child_samples", 5, 100),
        subsample=trial.suggest_float("subsample", 0.5, 1.0),
        subsample_freq=1,
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.3, 1.0),
        reg_alpha=trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        reg_lambda=trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        class_weight="balanced",
        n_jobs=-1,
        random_state=42,
        verbose=-1,
    )
    return _cv_macro_f1(lambda: lgb.LGBMClassifier(**params), X, y)


def tune_lgbm(X, y) -> dict:
    log.info("Tuning LightGBM (%d trials)...", N_TRIALS)
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(),
    )
    study.optimize(lambda t: _lgbm_objective(t, X, y), n_trials=N_TRIALS, show_progress_bar=False)
    log.info("LightGBM best macro_f1=%.4f", study.best_value)
    return study.best_params


# ── XGBoost ───────────────────────────────────────────────────────────────────

def _xgb_objective(trial: optuna.Trial, X, y) -> float:
    # Compute balanced class weights
    n = len(y)
    classes, counts = np.unique(y, return_counts=True)
    weights = {int(c): n / (len(classes) * cnt) for c, cnt in zip(classes, counts)}
    sample_weights = np.array([weights[yi] for yi in y])

    params = dict(
        objective="multi:softprob",
        num_class=3,
        n_estimators=trial.suggest_int("n_estimators", 300, 2000),
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        max_depth=trial.suggest_int("max_depth", 3, 10),
        min_child_weight=trial.suggest_int("min_child_weight", 1, 20),
        subsample=trial.suggest_float("subsample", 0.5, 1.0),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.3, 1.0),
        colsample_bylevel=trial.suggest_float("colsample_bylevel", 0.3, 1.0),
        gamma=trial.suggest_float("gamma", 0, 5),
        reg_alpha=trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        reg_lambda=trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        max_delta_step=trial.suggest_int("max_delta_step", 0, 5),
        tree_method="hist",
        n_jobs=-1,
        random_state=42,
        eval_metric="mlogloss",
    )
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    scores = []
    for tr, val in cv.split(X, y):
        sw = sample_weights[tr]
        m = xgb.XGBClassifier(**params, verbosity=0)
        m.fit(X.iloc[tr], y[tr], sample_weight=sw, verbose=False)
        preds = np.argmax(m.predict_proba(X.iloc[val]), axis=1)
        scores.append(f1_score(y[val], preds, average="macro", zero_division=0))
    return float(np.mean(scores))


def tune_xgb(X, y) -> dict:
    log.info("Tuning XGBoost (%d trials)...", N_TRIALS)
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(),
    )
    study.optimize(lambda t: _xgb_objective(t, X, y), n_trials=N_TRIALS, show_progress_bar=False)
    log.info("XGBoost best macro_f1=%.4f", study.best_value)
    return study.best_params


# ── CatBoost ──────────────────────────────────────────────────────────────────

def _cat_objective(trial: optuna.Trial, X, y) -> float:
    params = dict(
        iterations=trial.suggest_int("iterations", 300, 2000),
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        depth=trial.suggest_int("depth", 4, 10),
        l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 1, 30, log=True),
        border_count=trial.suggest_int("border_count", 32, 254),
        random_strength=trial.suggest_float("random_strength", 0.1, 10.0),
        bagging_temperature=trial.suggest_float("bagging_temperature", 0, 5),
        min_data_in_leaf=trial.suggest_int("min_data_in_leaf", 1, 50),
        auto_class_weights=trial.suggest_categorical("auto_class_weights", ["Balanced", "SqrtBalanced"]),
        loss_function="MultiClass",
        eval_metric="TotalF1",
        random_seed=42,
        verbose=False,
        thread_count=-1,
    )
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    scores = []
    for tr, val in cv.split(X, y):
        m = CatBoostClassifier(**params)
        m.fit(X.iloc[tr], y[tr], silent=True)
        preds = np.argmax(m.predict_proba(X.iloc[val]), axis=1)
        scores.append(f1_score(y[val], preds, average="macro", zero_division=0))
    return float(np.mean(scores))


def tune_catboost(X, y) -> dict:
    log.info("Tuning CatBoost (%d trials)...", N_TRIALS)
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(),
    )
    study.optimize(lambda t: _cat_objective(t, X, y), n_trials=N_TRIALS, show_progress_bar=False)
    log.info("CatBoost best macro_f1=%.4f", study.best_value)
    return study.best_params


# ── Full CV OOF generation with best params ───────────────────────────────────

def _full_cv_oof(build_fn, X, y, n_splits=CV_FOLDS) -> np.ndarray:
    """Generate OOF predictions via stratified K-fold."""
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    oof = np.zeros((len(X), 3))
    for fold, (tr, val) in enumerate(cv.split(X, y)):
        m = build_fn()
        m.fit(X.iloc[tr], y[tr])
        oof[val] = m.predict_proba(X.iloc[val])
        log.info("  fold %d done", fold + 1)
    return oof


# ── Main ──────────────────────────────────────────────────────────────────────

def run(n_trials: int = N_TRIALS) -> dict:
    global N_TRIALS
    N_TRIALS = n_trials

    log.info("=" * 60)
    log.info("Pipeline 2: Optuna Big Three")
    log.info("=" * 60)

    X, y = load_train_data()
    X_tv, X_hold, y_tv, y_hold = make_holdout(X, y)

    # Load holdout indices saved by P01 (or reuse same split)
    np.save(ARTIFACTS / "y_holdout.npy", y_hold)

    results = {}
    best_params = {}

    for name, tune_fn, build_fn_factory in [
        ("lgbm", tune_lgbm, lambda p: lgb.LGBMClassifier(
            objective="multiclass", num_class=3, class_weight="balanced",
            n_jobs=-1, random_state=42, verbose=-1, **p)),
        ("xgb", tune_xgb, lambda p: xgb.XGBClassifier(
            objective="multi:softprob", num_class=3, tree_method="hist",
            n_jobs=-1, random_state=42, verbosity=0, **p)),
        ("catboost", tune_catboost, lambda p: CatBoostClassifier(
            loss_function="MultiClass", random_seed=42, verbose=False,
            thread_count=-1, **p)),
    ]:
        params = tune_fn(X_tv, y_tv)
        best_params[name] = params

        log.info("Generating 5-fold OOF for %s...", name)
        oof = _full_cv_oof(lambda p=params, f=build_fn_factory: f(p), X_tv, y_tv)
        save_oof(f"p02_{name}", oof)

        cv_res = evaluate_proba(f"P02_{name}_oof", y_tv, oof)
        save_result(cv_res)

        # Holdout evaluation
        m = build_fn_factory(params)
        m.fit(X_tv, y_tv)
        hold_proba = m.predict_proba(X_hold)
        hold_res = evaluate_proba(f"P02_{name}_holdout", y_hold, hold_proba)
        save_result(hold_res)
        np.save(ARTIFACTS / f"oof_p02_{name}_holdout.npy", hold_proba)

        results[name] = hold_res

    # Persist best params
    (ARTIFACTS / "best_params_p02.json").write_text(json.dumps(best_params, indent=2))
    log.info("Best params saved.")

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-trials", type=int, default=N_TRIALS)
    args = parser.parse_args()
    run(n_trials=args.n_trials)
