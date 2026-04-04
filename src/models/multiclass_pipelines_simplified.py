"""
Simplified and feature-parametric multiclass pipelines.

New model types
---------------
Ks  — LightGBM with tighter regularisation (num_leaves=31, reg_lambda=3.0)
RF  — RandomForestClassifier (no boosting, high bias — good overfitting check)
HGB — HistGradientBoostingClassifier (sklearn native, well-regularised by default)
LR  — LogisticRegression + StandardScaler (linear baseline)

Existing architectures (K / L / M) are also re-run here with reduced feature sets
so we get apples-to-apples comparisons across all feature budgets.

Feature subsets
---------------
  all    — all S1_FEATURES present in the parquet (~139)
  top100 — SHAP top-100
  top75  — SHAP top-75
  top50  — SHAP top-50
  top25  — SHAP top-25

Pipeline names encode both model and feature set, e.g. "Ks_top50", "K_top25".
Results are written to models/artifacts/results_simplified.csv.

Usage:
    uv run python main_simplified.py                         # all models × all subsets
    uv run python main_simplified.py --models Ks RF          # subset of models
    uv run python main_simplified.py --features top50 top25  # subset of feature sets
    uv run python main_simplified.py --models K --features top100 top75 top50 top25
"""
from __future__ import annotations

import json
from itertools import product

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from catboost import CatBoostClassifier, Pool
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.churn.multiclass_classifier import build_lgbm_multiclass, build_xgb_multiclass
from src.models.multiclass_evaluate import evaluate_multiclass
from src.models.shap_feature_ranking import load_feature_lists
from src.models.train import (
    CAT_FEATURES_S1,
    S1_FEATURES,
    load_feature_matrix,
    make_splits,
    safe_features,
)
from src.utils.helpers import root_path
from src.utils.logger import get_logger

log = get_logger(__name__)

_ARTIFACTS = root_path() / "models" / "artifacts"
_TRAINED   = root_path() / "models" / "trained"

# ── Simplified model builders ──────────────────────────────────────────────────

def build_lgbm_simple() -> lgb.LGBMClassifier:
    """LGBM with tighter regularisation — fewer leaves, stronger L2."""
    return lgb.LGBMClassifier(
        objective="multiclass",
        num_class=3,
        n_estimators=1000,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=60,
        subsample=0.80,
        subsample_freq=1,
        colsample_bytree=0.70,
        reg_alpha=0.5,
        reg_lambda=3.0,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42,
        verbose=-1,
    )


def build_rf() -> RandomForestClassifier:
    """Random Forest — no boosting, high-bias check for overfitting."""
    return RandomForestClassifier(
        n_estimators=300,
        max_depth=15,
        min_samples_leaf=25,
        max_features="sqrt",
        class_weight="balanced",
        n_jobs=-1,
        random_state=42,
    )


def build_hgb() -> HistGradientBoostingClassifier:
    """sklearn HistGradientBoosting — fast, well-regularised, handles NaN natively."""
    return HistGradientBoostingClassifier(
        max_iter=500,
        max_leaf_nodes=31,
        min_samples_leaf=25,
        l2_regularization=1.0,
        learning_rate=0.05,
        class_weight="balanced",
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
        random_state=42,
    )


def build_lr() -> Pipeline:
    """Logistic Regression with StandardScaler — linear sanity baseline."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            C=1.0,
            max_iter=2000,
            class_weight="balanced",
            solver="saga",
            multi_class="multinomial",
            n_jobs=-1,
            random_state=42,
        )),
    ])


# Registry: name → builder function (None = CatBoost handled separately)
_BUILDERS: dict[str, callable] = {
    "K":   build_lgbm_multiclass,   # existing, now parametric by features
    "L":   build_xgb_multiclass,    # existing, now parametric by features
    "Ks":  build_lgbm_simple,
    "RF":  build_rf,
    "HGB": build_hgb,
    "LR":  build_lr,
}
_CATBOOST_MODELS = {"M"}
ALL_MODELS = sorted(_BUILDERS.keys()) + sorted(_CATBOOST_MODELS)


# ── Generic CV (LGBM / XGB / sklearn-compatible) ───────────────────────────────

def _fit_model(model, X_tr, y_tr, X_val, y_val) -> None:
    if isinstance(model, lgb.LGBMClassifier):
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            eval_names=["val"],
            callbacks=[
                lgb.log_evaluation(period=200),
                lgb.early_stopping(stopping_rounds=100, verbose=False),
            ],
        )
    elif isinstance(model, xgb.XGBClassifier):
        model.set_params(early_stopping_rounds=100)
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=200)
    else:
        model.fit(X_tr, y_tr)


def _mc_cv(
    build_fn: callable,
    X: pd.DataFrame,
    y_3: np.ndarray,
    features: list[str],
    n_splits: int = 5,
) -> np.ndarray:
    """Stratified K-fold CV returning OOF probability matrix (n_samples, 3)."""
    feat = safe_features(X, features)
    cv   = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    oof  = np.zeros((len(X), 3))

    log.info("  Running %d-fold CV on %d features", n_splits, len(feat))
    for fold, (tr_idx, val_idx) in enumerate(cv.split(X, y_3)):
        Xtr  = X.iloc[tr_idx][feat]
        Xval = X.iloc[val_idx][feat]
        ytr  = y_3[tr_idx]
        yval = y_3[val_idx]

        model = build_fn()
        _fit_model(model, Xtr, ytr, Xval, yval)
        oof[val_idx] = model.predict_proba(Xval)

        log.info(
            "  Fold %d — macro F1: %.4f",
            fold + 1,
            f1_score(yval, np.argmax(oof[val_idx], axis=1),
                     average="macro", zero_division=0),
        )

    return oof


# ── CatBoost CV (needs Pool objects) ──────────────────────────────────────────

def _catboost_cv(
    X: pd.DataFrame,
    y_3: np.ndarray,
    features: list[str],
    n_splits: int = 5,
) -> np.ndarray:
    feat    = safe_features(X, features)
    cat_col = [c for c in CAT_FEATURES_S1 if c in feat]
    cv      = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    oof     = np.zeros((len(X), 3))

    log.info("  CatBoost %d-fold CV on %d features (%d cat)", n_splits, len(feat), len(cat_col))
    for fold, (tr_idx, val_idx) in enumerate(cv.split(X, y_3)):
        Xtr  = X.iloc[tr_idx][feat].copy()
        Xval = X.iloc[val_idx][feat].copy()
        for c in cat_col:
            Xtr[c]  = Xtr[c].fillna("unknown").astype(str)
            Xval[c] = Xval[c].fillna("unknown").astype(str)

        model = CatBoostClassifier(
            loss_function="MultiClass",
            classes_count=3,
            iterations=1000,
            learning_rate=0.05,
            depth=7,
            l2_leaf_reg=3.0,
            bagging_temperature=0.75,
            auto_class_weights="Balanced",
            eval_metric="TotalF1",
            random_seed=42,
            verbose=0,
            early_stopping_rounds=50,
        )
        model.fit(
            Pool(Xtr,  label=y_3[tr_idx],  cat_features=cat_col),
            eval_set=Pool(Xval, label=y_3[val_idx], cat_features=cat_col),
            use_best_model=True,
            verbose=0,
        )
        oof[val_idx] = model.predict_proba(
            Pool(Xval, cat_features=cat_col)
        )

        log.info(
            "  Fold %d — macro F1: %.4f",
            fold + 1,
            f1_score(y_3[val_idx], np.argmax(oof[val_idx], axis=1),
                     average="macro", zero_division=0),
        )

    return oof


# ── Single pipeline runner ─────────────────────────────────────────────────────

def run_pipeline(
    model_name: str,
    feature_key: str,
    X: pd.DataFrame,
    y_3: np.ndarray,
    feature_lists: dict[str, list[str]],
) -> dict:
    """Run one (model, feature_set) combination and return metrics dict."""
    name     = f"{model_name}_{feature_key}"
    features = feature_lists[feature_key]

    log.info("=== Pipeline %s (%d features) ===", name, len(features))

    if model_name in _CATBOOST_MODELS:
        oof = _catboost_cv(X, y_3, features)
    else:
        build_fn = _BUILDERS[model_name]
        oof = _mc_cv(build_fn, X, y_3, features)

    _ARTIFACTS.mkdir(parents=True, exist_ok=True)
    np.save(_ARTIFACTS / f"oof_mc_{name}.npy", oof)

    result = evaluate_multiclass(name, y_3, oof)
    with open(_ARTIFACTS / f"results_{name}.json", "w") as fh:
        json.dump(result, fh, indent=2)

    return result


# ── Batch runner ───────────────────────────────────────────────────────────────

def run_simplified_suite(
    models: list[str] | None = None,
    feature_keys: list[str] | None = None,
) -> pd.DataFrame:
    """Run all (model, feature_set) combinations and return ranked DataFrame.

    Args:
        models:       Subset of ALL_MODELS to run (default: all).
        feature_keys: Subset of feature_list keys to use
                      (default: ["all", "top100", "top75", "top50", "top25"]).
    """
    models       = models       or ALL_MODELS
    feature_keys = feature_keys or ["all", "top100", "top75", "top50", "top25"]

    # Validate
    unknown_m = set(models) - set(ALL_MODELS)
    if unknown_m:
        raise ValueError(f"Unknown model(s): {unknown_m}. Choose from {ALL_MODELS}")

    feature_lists = load_feature_lists()
    unknown_f = set(feature_keys) - set(feature_lists)
    if unknown_f:
        raise ValueError(f"Unknown feature key(s): {unknown_f}. Available: {list(feature_lists)}")

    X, y = load_feature_matrix()
    X_train, _, _, y_train, _, _ = make_splits(X, y)
    y_3 = y_train.values

    results = []
    for model_name, fkey in product(models, feature_keys):
        try:
            results.append(run_pipeline(model_name, fkey, X_train, y_3, feature_lists))
        except Exception as e:
            log.error("Pipeline %s_%s failed: %s", model_name, fkey, e, exc_info=True)
            results.append({"pipeline": f"{model_name}_{fkey}", "error": str(e)})

    df = pd.DataFrame(results)
    for sort_col in ["weighted_f1_3class", "macro_f1_3class"]:
        if sort_col in df.columns:
            df = df.sort_values(sort_col, ascending=False)
            break

    _ARTIFACTS.mkdir(parents=True, exist_ok=True)
    df.to_csv(_ARTIFACTS / "results_simplified.csv", index=False)
    log.info("Results saved → models/artifacts/results_simplified.csv")
    return df
