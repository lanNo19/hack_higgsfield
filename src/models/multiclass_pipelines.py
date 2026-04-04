"""
Alternative churn pipelines — direct 3-class models and fixed cascade.

Pipeline K — LightGBM native multiclass (3 classes, no cascade)
Pipeline L — XGBoost native multiclass
Pipeline M — CatBoost native multiclass
Pipeline N — Fixed cascade: LightGBM S1 + LightGBM S2 (all features) + joint thresholds

Why this matters vs A–F:
  - K/L/M skip the cascade entirely; a single model optimises all 3 classes jointly.
    No errors propagate from Stage 1 to Stage 2.
  - N keeps the cascade but fixes two bugs from A–F:
      1. S2 now sees ALL features, not just T_FEATURES (vol_churn ≠ payment failure).
      2. Thresholds are found via joint grid-search maximising 3-class macro F1,
         not independently optimising the binary S1 F1 (which was worse than baseline).

Usage:
    uv run python main_multiclass.py              # K L M N
    uv run python main_multiclass.py --pipeline K # single
"""
from __future__ import annotations

import json

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold

from src.churn.multiclass_classifier import (
    build_catboost_multiclass,
    build_lgbm_multiclass,
    build_lgbm_s2_allfeat,
    build_xgb_multiclass,
)
from src.models.multiclass_evaluate import evaluate_fixed_cascade, evaluate_multiclass
from src.models.train import (
    CAT_FEATURES_S1,
    S1_FEATURES,
    load_feature_matrix,
    make_labels,
    make_splits,
    safe_features,
)
from src.utils.helpers import root_path
from src.utils.logger import get_logger

log = get_logger(__name__)

_ARTIFACTS = root_path() / "models" / "artifacts"
_TRAINED   = root_path() / "models" / "trained"


# ── OOF persistence ────────────────────────────────────────────────────────────

def _save_oof_mc(name: str, oof_proba: np.ndarray) -> None:
    """Save (n, 3) multiclass OOF probabilities."""
    _ARTIFACTS.mkdir(parents=True, exist_ok=True)
    np.save(_ARTIFACTS / f"oof_mc_{name}.npy", oof_proba)


def _save_oof_cascade(name: str, oof_s1: np.ndarray, oof_s2: np.ndarray) -> None:
    _ARTIFACTS.mkdir(parents=True, exist_ok=True)
    np.save(_ARTIFACTS / f"oof_s1_{name}.npy", oof_s1)
    np.save(_ARTIFACTS / f"oof_s2_{name}.npy", oof_s2)


# ── Shared fit helpers ─────────────────────────────────────────────────────────

def _fit_lgbm_mc(model: lgb.LGBMClassifier,
                 X_tr, y_tr, X_val, y_val,
                 log_period: int = 100) -> None:
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        eval_names=["val"],
        callbacks=[
            lgb.log_evaluation(period=log_period),
            lgb.early_stopping(stopping_rounds=100, verbose=True),
        ],
    )


def _fit_xgb_mc(model: xgb.XGBClassifier,
                X_tr, y_tr, X_val, y_val,
                log_period: int = 100) -> None:
    model.set_params(early_stopping_rounds=100)
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=log_period)


# ══════════════════════════════════════════════════════════════════════════════
# Generic multiclass CV loop (K and L)
# ══════════════════════════════════════════════════════════════════════════════

def _mc_cv(
    build_model,
    X: pd.DataFrame,
    y_3: np.ndarray,
    n_splits: int = 5,
    log_period: int = 100,
) -> np.ndarray:
    """Stratified K-fold CV returning OOF probability matrix (n, 3).

    Handles LightGBM and XGBoost via their respective fit helpers.
    Falls back to plain .fit() for other sklearn-compatible models.
    """
    feat  = safe_features(X, S1_FEATURES)
    cv    = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    oof   = np.zeros((len(X), 3))

    for fold, (tr_idx, val_idx) in enumerate(cv.split(X, y_3)):
        Xtr  = X.iloc[tr_idx][feat]
        Xval = X.iloc[val_idx][feat]
        ytr  = y_3[tr_idx]
        yval = y_3[val_idx]

        model = build_model()

        if isinstance(model, lgb.LGBMClassifier):
            _fit_lgbm_mc(model, Xtr, ytr, Xval, yval, log_period)
        elif isinstance(model, xgb.XGBClassifier):
            _fit_xgb_mc(model, Xtr, ytr, Xval, yval, log_period)
        else:
            model.fit(Xtr, ytr)

        oof[val_idx] = model.predict_proba(Xval)

        fold_preds = np.argmax(oof[val_idx], axis=1)
        from sklearn.metrics import f1_score
        log.info(
            "Fold %d — val macro F1: %.4f",
            fold + 1,
            f1_score(yval, fold_preds, average="macro", zero_division=0),
        )

    return oof


# ══════════════════════════════════════════════════════════════════════════════
# Pipeline K — LightGBM multiclass
# ══════════════════════════════════════════════════════════════════════════════

def run_pipeline_K(X: pd.DataFrame, y_3: np.ndarray) -> dict:
    log.info("=== Pipeline K: LightGBM native 3-class ===")
    oof = _mc_cv(build_lgbm_multiclass, X, y_3)
    _save_oof_mc("K", oof)
    return evaluate_multiclass("K", y_3, oof)


# ══════════════════════════════════════════════════════════════════════════════
# Pipeline L — XGBoost multiclass
# ══════════════════════════════════════════════════════════════════════════════

def run_pipeline_L(X: pd.DataFrame, y_3: np.ndarray) -> dict:
    log.info("=== Pipeline L: XGBoost native 3-class ===")
    oof = _mc_cv(build_xgb_multiclass, X, y_3)
    _save_oof_mc("L", oof)
    return evaluate_multiclass("L", y_3, oof)


# ══════════════════════════════════════════════════════════════════════════════
# Pipeline M — CatBoost multiclass (needs Pool + cat features)
# ══════════════════════════════════════════════════════════════════════════════

def _catboost_mc_cv(
    X: pd.DataFrame,
    y_3: np.ndarray,
    n_splits: int = 5,
) -> np.ndarray:
    """CatBoost multiclass CV — requires Pool objects for categorical handling."""
    feat    = safe_features(X, S1_FEATURES)
    cat_col = [c for c in CAT_FEATURES_S1 if c in feat]
    cv      = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    oof     = np.zeros((len(X), 3))

    for fold, (tr_idx, val_idx) in enumerate(cv.split(X, y_3)):
        Xtr  = X.iloc[tr_idx][feat].copy()
        Xval = X.iloc[val_idx][feat].copy()
        for c in cat_col:
            Xtr[c]  = Xtr[c].fillna("unknown").astype(str)
            Xval[c] = Xval[c].fillna("unknown").astype(str)

        tr_pool  = Pool(Xtr,  label=y_3[tr_idx],  cat_features=cat_col)
        val_pool = Pool(Xval, label=y_3[val_idx], cat_features=cat_col)

        model = build_catboost_multiclass()
        model.fit(tr_pool, eval_set=val_pool, use_best_model=True, verbose=100)
        oof[val_idx] = model.predict_proba(val_pool)

        fold_preds = np.argmax(oof[val_idx], axis=1)
        from sklearn.metrics import f1_score
        log.info(
            "Fold %d — val macro F1: %.4f",
            fold + 1,
            f1_score(y_3[val_idx], fold_preds, average="macro", zero_division=0),
        )

    return oof


def run_pipeline_M(X: pd.DataFrame, y_3: np.ndarray) -> dict:
    log.info("=== Pipeline M: CatBoost native 3-class ===")
    oof = _catboost_mc_cv(X, y_3)
    _save_oof_mc("M", oof)
    return evaluate_multiclass("M", y_3, oof)


# ══════════════════════════════════════════════════════════════════════════════
# Pipeline N — Fixed cascade: all features in S2 + joint threshold search
#
# Fixes two bugs vs A–F:
#   1. S2 uses S1_FEATURES (all features), not just T_FEATURES
#   2. Thresholds found via joint grid-search maximising 3-class macro F1
# ══════════════════════════════════════════════════════════════════════════════

def _fixed_cascade_cv(
    X: pd.DataFrame,
    y_binary: np.ndarray,
    y_volInv: np.ndarray,
    n_splits: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """Two-stage CV where Stage 2 uses ALL features (not just T_FEATURES)."""
    feat = safe_features(X, S1_FEATURES)
    cv   = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    oof_s1 = np.zeros(len(X))
    oof_s2 = np.zeros(len(X))

    for fold, (tr_idx, val_idx) in enumerate(cv.split(X, y_binary)):
        Xtr   = X.iloc[tr_idx][feat]
        Xval  = X.iloc[val_idx][feat]
        y_tr  = y_binary[tr_idx]
        y_val = y_binary[val_idx]

        # Stage 1 — same as Pipeline A
        s1 = lgb.LGBMClassifier(
            n_estimators=1000, learning_rate=0.05, num_leaves=95,
            min_child_samples=30, subsample=0.80, subsample_freq=1,
            colsample_bytree=0.70, reg_alpha=0.1, reg_lambda=1.0,
            is_unbalance=True, n_jobs=-1, random_state=42, verbose=-1,
        )
        _fit_lgbm_mc(s1, Xtr, y_tr, Xval, y_val)
        oof_s1[val_idx] = s1.predict_proba(Xval)[:, 1]

        # Stage 2 — ALL features on churned subset only
        churn_tr  = tr_idx[y_binary[tr_idx] == 1]
        churn_val = val_idx[y_binary[val_idx] == 1]
        if len(churn_tr) > 10 and len(churn_val) > 0:
            s2 = build_lgbm_s2_allfeat()
            Xtr2  = X.iloc[churn_tr][feat]
            Xval2 = X.iloc[churn_val][feat]
            _fit_lgbm_mc(s2, Xtr2, y_volInv[churn_tr], Xval2, y_volInv[churn_val],
                         log_period=100)
            oof_s2[churn_val] = s2.predict_proba(Xval2)[:, 1]

        log.info(
            "Fold %d — S1 PR-AUC: %.4f",
            fold + 1,
            average_precision_score(y_val, oof_s1[val_idx]),
        )

    return oof_s1, oof_s2


def run_pipeline_N(
    X: pd.DataFrame,
    y_binary: np.ndarray,
    y_volInv: np.ndarray,
) -> dict:
    log.info("=== Pipeline N: Fixed cascade (all features in S2 + joint thresholds) ===")
    oof_s1, oof_s2 = _fixed_cascade_cv(X, y_binary, y_volInv)
    _save_oof_cascade("N", oof_s1, oof_s2)
    return evaluate_fixed_cascade("N", y_binary, y_volInv, oof_s1, oof_s2)


# ══════════════════════════════════════════════════════════════════════════════
# Unified runner
# ══════════════════════════════════════════════════════════════════════════════

_MC_RUNNERS   = {"K", "L", "M"}   # direct multiclass — take (X, y_3)
_CASC_RUNNERS = {"N"}             # fixed cascade — take (X, y_binary, y_volInv)
_ALL_RUNNERS  = _MC_RUNNERS | _CASC_RUNNERS


def run_mc_pipeline(name: str) -> dict:
    """Load data, run the named pipeline, save result JSON, return metrics."""
    assert name in _ALL_RUNNERS, f"Unknown pipeline '{name}'. Choose from: {sorted(_ALL_RUNNERS)}"

    X, y = load_feature_matrix()
    X_train, _, _, y_train, _, _ = make_splits(X, y)

    _ARTIFACTS.mkdir(parents=True, exist_ok=True)

    if name in _MC_RUNNERS:
        y_3 = y_train.values  # already 0/1/2 from _LABEL_MAP
        runners = {"K": run_pipeline_K, "L": run_pipeline_L, "M": run_pipeline_M}
        result = runners[name](X_train, y_3)
    else:
        y_binary, y_volInv = make_labels(y_train)
        result = run_pipeline_N(X_train, y_binary, y_volInv)

    with open(_ARTIFACTS / f"results_{name}.json", "w") as f:
        json.dump(result, f, indent=2)
    log.info("Results saved → models/artifacts/results_%s.json", name)
    return result


def run_all_mc_pipelines(pipelines: list[str] | None = None) -> pd.DataFrame:
    """Run all (or a subset of) K–N pipelines, return ranked DataFrame."""
    names = pipelines or sorted(_ALL_RUNNERS)
    results = []
    for name in names:
        try:
            results.append(run_mc_pipeline(name))
        except Exception as e:
            log.error("Pipeline %s failed: %s", name, e, exc_info=True)
            results.append({"pipeline": name, "error": str(e)})

    df = pd.DataFrame(results)
    # Sort by weighted F1 — the submission metric
    for sort_col in ["weighted_f1_3class", "weighted_f1_joint_thr", "macro_f1_3class"]:
        if sort_col in df.columns:
            df = df.sort_values(sort_col, ascending=False)
            break

    _ARTIFACTS.mkdir(parents=True, exist_ok=True)
    df.to_csv(_ARTIFACTS / "results_multiclass.csv", index=False)
    return df
