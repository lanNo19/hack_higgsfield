"""
V2 multiclass pipelines — same K / L / M / N structure as multiclass_pipelines.py
but using the v2 feature matrix (targeted new features, noise removed).

Load v2 features first:
    uv run python run_features_v2.py

Then run:
    uv run python main_multiclass_v2.py
"""
from __future__ import annotations

import json

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import average_precision_score, f1_score
from sklearn.model_selection import StratifiedKFold

from src.churn.multiclass_classifier import (
    build_catboost_multiclass,
    build_lgbm_multiclass,
    build_lgbm_s2_allfeat,
    build_xgb_multiclass,
)
from src.models.multiclass_evaluate import evaluate_fixed_cascade, evaluate_multiclass
from src.models.train import CAT_FEATURES_S1, make_labels, make_splits
from src.models.train_v2 import (
    S1_FEATURES_V2,
    T_FEATURES_V2,
    load_feature_matrix_v2,
    safe_features,
)
from src.utils.helpers import root_path
from src.utils.logger import get_logger

log = get_logger(__name__)

_ARTIFACTS = root_path() / "models" / "artifacts"
_TRAINED   = root_path() / "models" / "trained"


# ── OOF helpers ───────────────────────────────────────────────────────────────

def _save_oof_mc(name: str, oof: np.ndarray) -> None:
    _ARTIFACTS.mkdir(parents=True, exist_ok=True)
    np.save(_ARTIFACTS / f"oof_mc_{name}.npy", oof)


def _save_oof_cascade(name: str, s1: np.ndarray, s2: np.ndarray) -> None:
    _ARTIFACTS.mkdir(parents=True, exist_ok=True)
    np.save(_ARTIFACTS / f"oof_s1_{name}.npy", s1)
    np.save(_ARTIFACTS / f"oof_s2_{name}.npy", s2)


# ── Fit helpers ───────────────────────────────────────────────────────────────

def _fit_lgbm(model: lgb.LGBMClassifier, Xtr, ytr, Xvl, yvl,
              log_period: int = 100) -> None:
    model.fit(
        Xtr, ytr,
        eval_set=[(Xvl, yvl)],
        eval_names=["val"],
        callbacks=[
            lgb.log_evaluation(period=log_period),
            lgb.early_stopping(stopping_rounds=100, verbose=True),
        ],
    )


def _fit_xgb(model: xgb.XGBClassifier, Xtr, ytr, Xvl, yvl,
             log_period: int = 100) -> None:
    model.set_params(early_stopping_rounds=100)
    model.fit(Xtr, ytr, eval_set=[(Xvl, yvl)], verbose=log_period)


# ── Generic multiclass CV (K and L) ───────────────────────────────────────────

def _mc_cv_v2(build_model, X: pd.DataFrame, y_3: np.ndarray,
              n_splits: int = 5) -> np.ndarray:
    feat = safe_features(X, S1_FEATURES_V2)
    cv   = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    oof  = np.zeros((len(X), 3))

    for fold, (tr_idx, val_idx) in enumerate(cv.split(X, y_3)):
        Xtr  = X.iloc[tr_idx][feat]
        Xvl  = X.iloc[val_idx][feat]
        ytr  = y_3[tr_idx]
        yvl  = y_3[val_idx]

        model = build_model()
        if isinstance(model, lgb.LGBMClassifier):
            _fit_lgbm(model, Xtr, ytr, Xvl, yvl)
        elif isinstance(model, xgb.XGBClassifier):
            _fit_xgb(model, Xtr, ytr, Xvl, yvl)
        else:
            model.fit(Xtr, ytr)

        oof[val_idx] = model.predict_proba(Xvl)
        log.info(
            "Fold %d — val macro F1: %.4f", fold + 1,
            f1_score(yvl, np.argmax(oof[val_idx], axis=1),
                     average="macro", zero_division=0),
        )

    return oof


# ══════════════════════════════════════════════════════════════════════════════
# Pipeline Kv2 — LightGBM multiclass, v2 features
# ══════════════════════════════════════════════════════════════════════════════

def run_pipeline_Kv2(X: pd.DataFrame, y_3: np.ndarray) -> dict:
    log.info("=== Pipeline Kv2: LightGBM 3-class, v2 features ===")
    oof = _mc_cv_v2(build_lgbm_multiclass, X, y_3)
    _save_oof_mc("Kv2", oof)
    return evaluate_multiclass("Kv2", y_3, oof)


# ══════════════════════════════════════════════════════════════════════════════
# Pipeline Lv2 — XGBoost multiclass, v2 features
# ══════════════════════════════════════════════════════════════════════════════

def run_pipeline_Lv2(X: pd.DataFrame, y_3: np.ndarray) -> dict:
    log.info("=== Pipeline Lv2: XGBoost 3-class, v2 features ===")
    oof = _mc_cv_v2(build_xgb_multiclass, X, y_3)
    _save_oof_mc("Lv2", oof)
    return evaluate_multiclass("Lv2", y_3, oof)


# ══════════════════════════════════════════════════════════════════════════════
# Pipeline Mv2 — CatBoost multiclass, v2 features
# ══════════════════════════════════════════════════════════════════════════════

def _catboost_mc_cv_v2(X: pd.DataFrame, y_3: np.ndarray,
                       n_splits: int = 5) -> np.ndarray:
    feat    = safe_features(X, S1_FEATURES_V2)
    cat_col = [c for c in CAT_FEATURES_S1 if c in feat]
    cv      = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    oof     = np.zeros((len(X), 3))

    for fold, (tr_idx, val_idx) in enumerate(cv.split(X, y_3)):
        Xtr  = X.iloc[tr_idx][feat].copy()
        Xvl  = X.iloc[val_idx][feat].copy()
        for c in cat_col:
            Xtr[c] = Xtr[c].fillna("unknown").astype(str)
            Xvl[c] = Xvl[c].fillna("unknown").astype(str)

        tr_pool  = Pool(Xtr,  label=y_3[tr_idx],  cat_features=cat_col)
        val_pool = Pool(Xvl, label=y_3[val_idx], cat_features=cat_col)

        model = build_catboost_multiclass()
        model.fit(tr_pool, eval_set=val_pool, use_best_model=True, verbose=100)
        oof[val_idx] = model.predict_proba(val_pool)
        log.info(
            "Fold %d — val macro F1: %.4f", fold + 1,
            f1_score(y_3[val_idx], np.argmax(oof[val_idx], axis=1),
                     average="macro", zero_division=0),
        )

    return oof


def run_pipeline_Mv2(X: pd.DataFrame, y_3: np.ndarray) -> dict:
    log.info("=== Pipeline Mv2: CatBoost 3-class, v2 features ===")
    oof = _catboost_mc_cv_v2(X, y_3)
    _save_oof_mc("Mv2", oof)
    return evaluate_multiclass("Mv2", y_3, oof)


# ══════════════════════════════════════════════════════════════════════════════
# Pipeline Nv2 — Fixed cascade with v2 features in both stages
# ══════════════════════════════════════════════════════════════════════════════

def _fixed_cascade_cv_v2(X: pd.DataFrame, y_binary: np.ndarray,
                          y_volInv: np.ndarray, n_splits: int = 5,
                          ) -> tuple[np.ndarray, np.ndarray]:
    """Cascade where BOTH stages use v2 features (T_FEATURES_V2 for S2)."""
    s1_feat = safe_features(X, S1_FEATURES_V2)
    t_feat  = safe_features(X, T_FEATURES_V2)

    cv     = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    oof_s1 = np.zeros(len(X))
    oof_s2 = np.zeros(len(X))

    for fold, (tr_idx, val_idx) in enumerate(cv.split(X, y_binary)):
        Xtr  = X.iloc[tr_idx][s1_feat]
        Xvl  = X.iloc[val_idx][s1_feat]
        y_tr = y_binary[tr_idx]
        y_vl = y_binary[val_idx]

        s1 = lgb.LGBMClassifier(
            n_estimators=1000, learning_rate=0.05, num_leaves=95,
            min_child_samples=30, subsample=0.80, subsample_freq=1,
            colsample_bytree=0.70, reg_alpha=0.1, reg_lambda=1.0,
            is_unbalance=True, n_jobs=-1, random_state=42, verbose=-1,
        )
        _fit_lgbm(s1, Xtr, y_tr, Xvl, y_vl)
        oof_s1[val_idx] = s1.predict_proba(Xvl)[:, 1]

        churn_tr  = tr_idx[y_binary[tr_idx] == 1]
        churn_val = val_idx[y_binary[val_idx] == 1]
        if len(churn_tr) > 10 and len(churn_val) > 0:
            s2    = build_lgbm_s2_allfeat()
            Xtr2  = X.iloc[churn_tr][t_feat]
            Xvl2  = X.iloc[churn_val][t_feat]
            _fit_lgbm(s2, Xtr2, y_volInv[churn_tr], Xvl2, y_volInv[churn_val])
            oof_s2[churn_val] = s2.predict_proba(Xvl2)[:, 1]

        log.info("Fold %d — S1 PR-AUC: %.4f", fold + 1,
                 average_precision_score(y_vl, oof_s1[val_idx]))

    return oof_s1, oof_s2


def run_pipeline_Nv2(X: pd.DataFrame, y_binary: np.ndarray,
                      y_volInv: np.ndarray) -> dict:
    log.info("=== Pipeline Nv2: Fixed cascade, v2 features + joint thresholds ===")
    oof_s1, oof_s2 = _fixed_cascade_cv_v2(X, y_binary, y_volInv)
    _save_oof_cascade("Nv2", oof_s1, oof_s2)
    return evaluate_fixed_cascade("Nv2", y_binary, y_volInv, oof_s1, oof_s2)


# ══════════════════════════════════════════════════════════════════════════════
# Unified runner
# ══════════════════════════════════════════════════════════════════════════════

_MC_RUNNERS   = {"Kv2", "Lv2", "Mv2"}
_CASC_RUNNERS = {"Nv2"}
_ALL_RUNNERS  = _MC_RUNNERS | _CASC_RUNNERS


def run_mc_pipeline_v2(name: str) -> dict:
    """Load v2 data, run the named pipeline, save result JSON, return metrics."""
    assert name in _ALL_RUNNERS, (
        f"Unknown pipeline '{name}'. Choose from: {sorted(_ALL_RUNNERS)}"
    )

    X, y = load_feature_matrix_v2()
    X_train, _, _, y_train, _, _ = make_splits(X, y)
    _ARTIFACTS.mkdir(parents=True, exist_ok=True)

    if name in _MC_RUNNERS:
        y_3 = y_train.values
        runners = {
            "Kv2": run_pipeline_Kv2,
            "Lv2": run_pipeline_Lv2,
            "Mv2": run_pipeline_Mv2,
        }
        result = runners[name](X_train, y_3)
    else:
        y_binary, y_volInv = make_labels(y_train)
        result = run_pipeline_Nv2(X_train, y_binary, y_volInv)

    out_path = _ARTIFACTS / f"results_{name}.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    log.info("Results saved → %s", out_path)
    return result


def run_all_mc_pipelines_v2(pipelines: list[str] | None = None) -> pd.DataFrame:
    names   = pipelines or sorted(_ALL_RUNNERS)
    results = []
    for name in names:
        try:
            results.append(run_mc_pipeline_v2(name))
        except Exception as e:
            log.error("Pipeline %s failed: %s", name, e, exc_info=True)
            results.append({"pipeline": name, "error": str(e)})

    df = pd.DataFrame(results)
    sort_col = "macro_f1_joint_thr" if "macro_f1_joint_thr" in df.columns else "macro_f1_3class"
    if sort_col in df.columns:
        df = df.sort_values(sort_col, ascending=False)

    _ARTIFACTS.mkdir(parents=True, exist_ok=True)
    df.to_csv(_ARTIFACTS / "results_multiclass_v2.csv", index=False)
    return df
