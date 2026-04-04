"""
Six churn-prediction pipelines (A–F).

All pipelines share:
  - Stratified train/val/test split with fixed random_state=42
  - 5-fold stratified CV on the train partition only
  - OOF predictions saved to models/artifacts/
  - Unified metrics via src.models.evaluate

Usage:
    from src.models.pipelines import run_pipeline, run_all_pipelines
    results = run_pipeline("A")        # single pipeline
    df      = run_all_pipelines()      # all six, returns ranked DataFrame
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from catboost import Pool
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold

from src.churn.classifier import (
    build_catboost_s1,
    build_lgbm_focal,
    build_lgbm_unbalanced,
    build_mlp_s1,
    build_stacking_s1,  # now returns StackingClassifier directly
    build_torch_mlp_s1,
)
from src.churn.thresholds import best_f1_threshold
from src.churn.voluntary_vs_involuntary import (
    build_catboost_s2,
    build_lgbm_s2,
    build_logreg_s2,
    build_voting_s2,
    build_xgb_s2,
)
from src.models.evaluate import evaluate
from src.models.train import (
    CAT_FEATURES_S1,
    CAT_FEATURES_S2,
    S1_FEATURES,
    T_FEATURES,
    _fit_with_eval,
    load_feature_matrix,
    make_labels,
    make_splits,
    run_two_stage_cv,
    safe_features,
)
from src.utils.helpers import root_path
from src.utils.logger import get_logger

log = get_logger(__name__)

_ARTIFACTS = root_path() / "models" / "artifacts"
_TRAINED   = root_path() / "models" / "trained"


# ── OOF persistence ────────────────────────────────────────────────────────────

def _save_oof(name: str, oof_s1: np.ndarray, oof_s2: np.ndarray) -> None:
    _ARTIFACTS.mkdir(parents=True, exist_ok=True)
    np.save(_ARTIFACTS / f"oof_s1_{name}.npy", oof_s1)
    np.save(_ARTIFACTS / f"oof_s2_{name}.npy", oof_s2)


# ══════════════════════════════════════════════════════════════════════════════
# Pipeline A — LightGBM (focal/unbalance) + XGBoost
# ══════════════════════════════════════════════════════════════════════════════

def run_pipeline_A(
    X: pd.DataFrame,
    y_binary: np.ndarray,
    y_volInv: np.ndarray,
) -> dict:
    log.info("=== Pipeline A: LightGBM (is_unbalance) + XGBoost ===")

    # Pass only churned-user subset so scale_pos_weight reflects the actual 50/50
    # vol vs invol balance in Stage 2, not the 75/25 overall class split
    y_churned = y_volInv[y_binary == 1]
    def s2(): return build_xgb_s2(y_churned)

    oof_s1, oof_s2 = run_two_stage_cv(build_lgbm_focal, s2, X, y_binary, y_volInv)
    _save_oof("A", oof_s1, oof_s2)
    return evaluate("A", y_binary, y_volInv, oof_s1, oof_s2)


# ══════════════════════════════════════════════════════════════════════════════
# Pipeline B — LightGBM + Logistic Regression
# ══════════════════════════════════════════════════════════════════════════════

def run_pipeline_B(
    X: pd.DataFrame,
    y_binary: np.ndarray,
    y_volInv: np.ndarray,
) -> dict:
    log.info("=== Pipeline B: LightGBM (is_unbalance) + Logistic Regression ===")
    oof_s1, oof_s2 = run_two_stage_cv(
        build_lgbm_unbalanced, build_logreg_s2, X, y_binary, y_volInv
    )
    _save_oof("B", oof_s1, oof_s2)
    return evaluate("B", y_binary, y_volInv, oof_s1, oof_s2)


# ══════════════════════════════════════════════════════════════════════════════
# Pipeline C — CatBoost + CatBoost
# ══════════════════════════════════════════════════════════════════════════════

def _catboost_cv(
    X: pd.DataFrame,
    y_binary: np.ndarray,
    y_volInv: np.ndarray,
    n_splits: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """CatBoost requires Pool objects, so it has its own CV loop."""
    s1_feat = safe_features(X, S1_FEATURES)
    t_feat  = safe_features(X, T_FEATURES)
    cat_s1  = [f for f in CAT_FEATURES_S1 if f in s1_feat]
    cat_s2  = [f for f in CAT_FEATURES_S2 if f in t_feat]

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    oof_s1 = np.zeros(len(X))
    oof_s2 = np.zeros(len(X))

    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y_binary)):
        Xtr  = X.iloc[train_idx][s1_feat].copy()
        Xval = X.iloc[val_idx][s1_feat].copy()
        for c in cat_s1:
            Xtr[c]  = Xtr[c].fillna("unknown").astype(str)
            Xval[c] = Xval[c].fillna("unknown").astype(str)

        train_pool = Pool(Xtr,  label=y_binary[train_idx], cat_features=cat_s1)
        val_pool   = Pool(Xval, label=y_binary[val_idx],   cat_features=cat_s1)

        s1 = build_catboost_s1()
        s1.fit(train_pool, eval_set=val_pool, use_best_model=True, verbose=100)
        oof_s1[val_idx] = s1.predict_proba(val_pool)[:, 1]

        churn_tr  = train_idx[y_binary[train_idx] == 1]
        churn_val = val_idx[y_binary[val_idx]   == 1]
        if len(churn_tr) > 10 and len(churn_val) > 0:
            Xtr2  = X.iloc[churn_tr][t_feat].copy()
            Xval2 = X.iloc[churn_val][t_feat].copy()
            for c in cat_s2:
                Xtr2[c]  = Xtr2[c].fillna("unknown").astype(str)
                Xval2[c] = Xval2[c].fillna("unknown").astype(str)
            tp = Pool(Xtr2,  label=y_volInv[churn_tr],  cat_features=cat_s2)
            vp = Pool(Xval2, label=y_volInv[churn_val], cat_features=cat_s2)
            s2 = build_catboost_s2(y_volInv[churn_tr])
            s2.fit(tp, eval_set=vp, use_best_model=True, verbose=50)
            oof_s2[churn_val] = s2.predict_proba(vp)[:, 1]

        log.info(
            "Fold %d — S1 PR-AUC: %.4f",
            fold + 1,
            average_precision_score(y_binary[val_idx], oof_s1[val_idx]),
        )

    return oof_s1, oof_s2


def run_pipeline_C(
    X: pd.DataFrame,
    y_binary: np.ndarray,
    y_volInv: np.ndarray,
) -> dict:
    log.info("=== Pipeline C: CatBoost + CatBoost ===")
    oof_s1, oof_s2 = _catboost_cv(X, y_binary, y_volInv)
    _save_oof("C", oof_s1, oof_s2)
    return evaluate("C", y_binary, y_volInv, oof_s1, oof_s2)


# ══════════════════════════════════════════════════════════════════════════════
# Pipeline D — Stacking Ensemble + VotingClassifier
# ══════════════════════════════════════════════════════════════════════════════

def run_pipeline_D(
    X: pd.DataFrame,
    y_binary: np.ndarray,
    y_volInv: np.ndarray,
) -> dict:
    log.info("=== Pipeline D: Stacking Ensemble + VotingClassifier ===")
    neg = (y_binary == 0).sum()
    pos = (y_binary == 1).sum()
    spw = neg / max(pos, 1)

    def s1(): return build_stacking_s1(scale_pos_weight=spw)
    y_churned = y_volInv[y_binary == 1]
    def s2(): return build_voting_s2(y_churned)

    oof_s1, oof_s2 = run_two_stage_cv(s1, s2, X, y_binary, y_volInv)
    _save_oof("D", oof_s1, oof_s2)
    return evaluate("D", y_binary, y_volInv, oof_s1, oof_s2)


# ══════════════════════════════════════════════════════════════════════════════
# Pipelines E & F — Three-level probability matrix
# ══════════════════════════════════════════════════════════════════════════════

def add_velocity_features(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    X["engagement_velocity"] = (
        (X.get("gens_last_7_days", pd.Series(0, index=X.index))
         - X.get("gens_first_7_days", pd.Series(0, index=X.index)))
        / (X.get("generation_span_days", pd.Series(0, index=X.index)) + 1)
    )
    if "avg_credit_cost_per_gen" in X.columns:
        X["credit_cost_decimal"] = X["avg_credit_cost_per_gen"] % 1
    return X


def add_synthetic_signal(X: pd.DataFrame) -> pd.DataFrame:
    """add_velocity_features + synthetic-distribution regularity features."""
    X = add_velocity_features(X)
    if "credit_cost_decimal" in X.columns:
        X["credit_regularity_score"] = (X["credit_cost_decimal"] == 0.0).astype(float)
    if "avg_inter_generation_hours" in X.columns:
        X["inter_gen_regularity"] = (
            (X["avg_inter_generation_hours"] % 1).abs() < 0.05
        ).astype(float)
    return X


def _blend_proba(proba_list: list[np.ndarray], weights: list[float]) -> np.ndarray:
    w = np.array(weights, dtype=float) / sum(weights)
    return sum(p * wi for p, wi in zip(proba_list, w))


def _three_level_cv(
    X: pd.DataFrame,
    y_binary: np.ndarray,
    y_volInv: np.ndarray,
    extra_s1_features: list[str],
    n_splits: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Three-model blend Stage 1 (LGBM + CatBoost + MLP) → Stage 2 (LGBM, T-features only).
    """
    s1_feat = safe_features(X, S1_FEATURES + extra_s1_features)
    t_feat  = safe_features(X, T_FEATURES)
    cat_cols = [c for c in CAT_FEATURES_S1 if c in s1_feat]

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    oof_p_churn = np.zeros(len(X))
    oof_p_invol = np.zeros(len(X))

    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y_binary)):
        Xtr  = X.iloc[train_idx][s1_feat]
        Xval = X.iloc[val_idx][s1_feat]
        y_tr = y_binary[train_idx]

        # CatBoost pool (needs string categoricals)
        Xtr_cat  = Xtr.copy()
        Xval_cat = Xval.copy()
        for c in cat_cols:
            Xtr_cat[c]  = Xtr_cat[c].fillna("unknown").astype(str)
            Xval_cat[c] = Xval_cat[c].fillna("unknown").astype(str)
        train_pool = Pool(Xtr_cat, label=y_tr,                cat_features=cat_cols)
        val_pool   = Pool(Xval_cat, label=y_binary[val_idx],  cat_features=cat_cols)

        # Train three Stage-1 models
        lgbm_m = build_lgbm_focal()
        cat_m  = build_catboost_s1()
        mlp_m  = build_mlp_s1()

        _fit_with_eval(lgbm_m, Xtr, y_tr, Xval, y_binary[val_idx])
        cat_m.fit(train_pool, eval_set=val_pool, use_best_model=True, verbose=100)
        mlp_m.fit(Xtr, y_tr)

        p_lgbm = lgbm_m.predict_proba(Xval)[:, 1]
        p_cat  = cat_m.predict_proba(val_pool)[:, 1]
        p_mlp  = mlp_m.predict_proba(Xval)[:, 1]
        oof_p_churn[val_idx] = _blend_proba([p_lgbm, p_cat, p_mlp], [0.4, 0.4, 0.2])

        # Stage 2: T-features only, churned subset
        churn_tr  = train_idx[y_binary[train_idx] == 1]
        churn_val = val_idx[y_binary[val_idx]   == 1]
        if len(churn_tr) > 10 and len(churn_val) > 0:
            s2 = build_lgbm_s2()
            _fit_with_eval(
                s2,
                X.iloc[churn_tr][t_feat], y_volInv[churn_tr],
                X.iloc[churn_val][t_feat], y_volInv[churn_val],
            )
            oof_p_invol[churn_val] = s2.predict_proba(
                X.iloc[churn_val][t_feat]
            )[:, 1]

        log.info(
            "Fold %d — S1 blend PR-AUC: %.4f",
            fold + 1,
            average_precision_score(y_binary[val_idx], oof_p_churn[val_idx]),
        )

    return oof_p_churn, oof_p_invol


def run_pipeline_E(
    X: pd.DataFrame,
    y_binary: np.ndarray,
    y_volInv: np.ndarray,
) -> dict:
    log.info("=== Pipeline E: Three-level probability matrix (real-data variant) ===")
    X = add_velocity_features(X)
    extra = [f for f in ["engagement_velocity", "credit_cost_decimal"] if f in X.columns]
    oof_s1, oof_s2 = _three_level_cv(X, y_binary, y_volInv, extra)
    _save_oof("E", oof_s1, oof_s2)
    return evaluate("E", y_binary, y_volInv, oof_s1, oof_s2)


def run_pipeline_F(
    X: pd.DataFrame,
    y_binary: np.ndarray,
    y_volInv: np.ndarray,
) -> dict:
    log.info("=== Pipeline F: Three-level probability matrix (synthetic-aware variant) ===")
    X = add_synthetic_signal(X)
    extra = [
        f for f in [
            "engagement_velocity", "credit_cost_decimal",
            "credit_regularity_score", "inter_gen_regularity",
        ] if f in X.columns
    ]
    oof_s1, oof_s2 = _three_level_cv(X, y_binary, y_volInv, extra)
    _save_oof("F", oof_s1, oof_s2)
    return evaluate("F", y_binary, y_volInv, oof_s1, oof_s2)


# ══════════════════════════════════════════════════════════════════════════════
# Pipeline G — Optuna-tuned LightGBM S1 + XGBoost S2
# ══════════════════════════════════════════════════════════════════════════════

def run_pipeline_G(
    X: pd.DataFrame,
    y_binary: np.ndarray,
    y_volInv: np.ndarray,
) -> dict:
    log.info("=== Pipeline G: Optuna-tuned LightGBM S1 + XGBoost S2 ===")
    from src.models.optuna_hpo import run_lgbm_study

    best_params = run_lgbm_study(X, y_binary)

    def s1(): return build_lgbm_focal(params=best_params)
    y_churned = y_volInv[y_binary == 1]
    def s2(): return build_xgb_s2(y_churned)

    oof_s1, oof_s2 = run_two_stage_cv(s1, s2, X, y_binary, y_volInv)
    _save_oof("G", oof_s1, oof_s2)
    return evaluate("G", y_binary, y_volInv, oof_s1, oof_s2)


# ══════════════════════════════════════════════════════════════════════════════
# Pipeline H — PyTorch MLP (BatchNorm + Dropout) S1 + LogReg S2
# ══════════════════════════════════════════════════════════════════════════════

def run_pipeline_H(
    X: pd.DataFrame,
    y_binary: np.ndarray,
    y_volInv: np.ndarray,
) -> dict:
    log.info("=== Pipeline H: PyTorch MLP S1 + Logistic Regression S2 ===")

    oof_s1, oof_s2 = run_two_stage_cv(
        build_torch_mlp_s1, build_logreg_s2, X, y_binary, y_volInv
    )
    _save_oof("H", oof_s1, oof_s2)
    return evaluate("H", y_binary, y_volInv, oof_s1, oof_s2)


# ══════════════════════════════════════════════════════════════════════════════
# Pipeline I — SMOTE oversampling + LightGBM S1 + XGBoost S2
# ══════════════════════════════════════════════════════════════════════════════

def _smote_cv(
    X: pd.DataFrame,
    y_binary: np.ndarray,
    y_volInv: np.ndarray,
    n_splits: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """CV loop that applies SMOTE inside each training fold (no leakage)."""
    from imblearn.over_sampling import SMOTE
    from src.utils.helpers import load_config

    k = load_config().get("smote", {}).get("k_neighbors", 5)
    s1_feat = safe_features(X, S1_FEATURES)
    t_feat  = safe_features(X, T_FEATURES)

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    oof_s1 = np.zeros(len(X))
    oof_s2 = np.zeros(len(X))

    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y_binary)):
        Xtr = X.iloc[train_idx][s1_feat].copy()
        Xvl = X.iloc[val_idx][s1_feat]
        ytr = y_binary[train_idx]
        yvl = y_binary[val_idx]

        # Fill NaN before SMOTE (it can't handle NaN)
        Xtr_filled = Xtr.fillna(Xtr.median())
        Xvl_filled = Xvl.fillna(Xtr.median())

        smote = SMOTE(k_neighbors=k, random_state=42)
        Xtr_res, ytr_res = smote.fit_resample(Xtr_filled, ytr)
        Xtr_res = pd.DataFrame(Xtr_res, columns=Xtr.columns)

        s1 = build_lgbm_focal()
        _fit_with_eval(s1, Xtr_res, ytr_res, Xvl_filled, yvl)
        oof_s1[val_idx] = s1.predict_proba(Xvl_filled)[:, 1]

        churn_tr  = train_idx[y_binary[train_idx] == 1]
        churn_val = val_idx[y_binary[val_idx] == 1]
        if len(churn_tr) > 10 and len(churn_val) > 0:
            y_churned = y_volInv[y_binary == 1]
            s2 = build_xgb_s2(y_churned)
            Xtr2 = X.iloc[churn_tr][t_feat].fillna(X.iloc[churn_tr][t_feat].median())
            Xvl2 = X.iloc[churn_val][t_feat].fillna(X.iloc[churn_tr][t_feat].median())
            _fit_with_eval(s2, Xtr2, y_volInv[churn_tr], Xvl2, y_volInv[churn_val])
            oof_s2[churn_val] = s2.predict_proba(Xvl2)[:, 1]

        log.info(
            "Fold %d — S1 PR-AUC: %.4f",
            fold + 1,
            average_precision_score(yvl, oof_s1[val_idx]),
        )

    return oof_s1, oof_s2


def run_pipeline_I(
    X: pd.DataFrame,
    y_binary: np.ndarray,
    y_volInv: np.ndarray,
) -> dict:
    log.info("=== Pipeline I: SMOTE + LightGBM S1 + XGBoost S2 ===")
    oof_s1, oof_s2 = _smote_cv(X, y_binary, y_volInv)
    _save_oof("I", oof_s1, oof_s2)
    return evaluate("I", y_binary, y_volInv, oof_s1, oof_s2)


# ══════════════════════════════════════════════════════════════════════════════
# Pipeline J — SHAP feature selection + CatBoost S1 + CatBoost S2
# ══════════════════════════════════════════════════════════════════════════════

def _shap_top_features(
    X: pd.DataFrame,
    y_binary: np.ndarray,
    top_k: int = 60,
) -> list[str]:
    """Train a quick LightGBM, compute SHAP values, return top-k feature names."""
    import shap
    from src.utils.helpers import load_config
    top_k = load_config().get("shap_selection", {}).get("top_k_features", top_k)

    feat = safe_features(X, S1_FEATURES)
    quick_model = build_lgbm_focal(params={"n_estimators": 300})
    quick_model.fit(X[feat].fillna(0), y_binary)

    explainer = shap.TreeExplainer(quick_model)
    shap_vals = explainer.shap_values(X[feat].fillna(0))
    # shap_values returns list [class0, class1] for binary; pick class1
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[1]
    importance = np.abs(shap_vals).mean(axis=0)
    ranked = sorted(zip(feat, importance), key=lambda x: x[1], reverse=True)
    selected = [f for f, _ in ranked[:top_k]]
    log.info("SHAP top-%d features selected (out of %d)", top_k, len(feat))
    return selected


def _catboost_cv_custom_features(
    X: pd.DataFrame,
    y_binary: np.ndarray,
    y_volInv: np.ndarray,
    s1_feat: list[str],
    n_splits: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """CatBoost CV with a custom feature list (e.g. SHAP-selected)."""
    t_feat  = safe_features(X, T_FEATURES)
    cat_s1  = [f for f in CAT_FEATURES_S1 if f in s1_feat]
    cat_s2  = [f for f in CAT_FEATURES_S2 if f in t_feat]

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    oof_s1 = np.zeros(len(X))
    oof_s2 = np.zeros(len(X))

    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y_binary)):
        Xtr  = X.iloc[train_idx][s1_feat].copy()
        Xval = X.iloc[val_idx][s1_feat].copy()
        for c in cat_s1:
            Xtr[c]  = Xtr[c].fillna("unknown").astype(str)
            Xval[c] = Xval[c].fillna("unknown").astype(str)

        from catboost import Pool
        train_pool = Pool(Xtr,  label=y_binary[train_idx], cat_features=cat_s1)
        val_pool   = Pool(Xval, label=y_binary[val_idx],   cat_features=cat_s1)

        s1 = build_catboost_s1()
        s1.fit(train_pool, eval_set=val_pool, use_best_model=True, verbose=100)
        oof_s1[val_idx] = s1.predict_proba(val_pool)[:, 1]

        churn_tr  = train_idx[y_binary[train_idx] == 1]
        churn_val = val_idx[y_binary[val_idx] == 1]
        if len(churn_tr) > 10 and len(churn_val) > 0:
            Xtr2  = X.iloc[churn_tr][t_feat].copy()
            Xval2 = X.iloc[churn_val][t_feat].copy()
            for c in cat_s2:
                Xtr2[c]  = Xtr2[c].fillna("unknown").astype(str)
                Xval2[c] = Xval2[c].fillna("unknown").astype(str)
            tp = Pool(Xtr2,  label=y_volInv[churn_tr],  cat_features=cat_s2)
            vp = Pool(Xval2, label=y_volInv[churn_val], cat_features=cat_s2)
            s2 = build_catboost_s2(y_volInv[churn_tr])
            s2.fit(tp, eval_set=vp, use_best_model=True, verbose=50)
            oof_s2[churn_val] = s2.predict_proba(vp)[:, 1]

        log.info(
            "Fold %d — S1 PR-AUC: %.4f",
            fold + 1,
            average_precision_score(y_binary[val_idx], oof_s1[val_idx]),
        )

    return oof_s1, oof_s2


def run_pipeline_J(
    X: pd.DataFrame,
    y_binary: np.ndarray,
    y_volInv: np.ndarray,
) -> dict:
    log.info("=== Pipeline J: SHAP feature selection + CatBoost S1 + CatBoost S2 ===")
    selected = _shap_top_features(X, y_binary)
    oof_s1, oof_s2 = _catboost_cv_custom_features(X, y_binary, y_volInv, selected)
    _save_oof("J", oof_s1, oof_s2)
    return evaluate("J", y_binary, y_volInv, oof_s1, oof_s2)


# ══════════════════════════════════════════════════════════════════════════════
# Unified entry point
# ══════════════════════════════════════════════════════════════════════════════

_RUNNERS = {
    "A": run_pipeline_A,
    "B": run_pipeline_B,
    "C": run_pipeline_C,
    "D": run_pipeline_D,
    "E": run_pipeline_E,
    "F": run_pipeline_F,
    "G": run_pipeline_G,
    "H": run_pipeline_H,
    "I": run_pipeline_I,
    "J": run_pipeline_J,
}


def run_pipeline(name: str) -> dict:
    """Load data, split, run the named pipeline on the train partition, return metrics."""
    assert name in _RUNNERS, f"Unknown pipeline '{name}'. Choose from: {list(_RUNNERS)}"

    X, y = load_feature_matrix()
    X_train, X_val, X_test, y_train, y_val, y_test = make_splits(X, y)
    y_bin_tr, y_inv_tr = make_labels(y_train)

    result = _RUNNERS[name](X_train, y_bin_tr, y_inv_tr)

    oof_s1 = np.load(_ARTIFACTS / f"oof_s1_{name}.npy")
    thr, _ = best_f1_threshold(y_bin_tr, oof_s1)
    result["val_threshold"] = round(thr, 3)

    _ARTIFACTS.mkdir(parents=True, exist_ok=True)
    with open(_ARTIFACTS / f"results_{name}.json", "w") as f:
        json.dump(result, f, indent=2)
    log.info("Results saved to models/artifacts/results_%s.json", name)

    return result


def run_all_pipelines(pipelines: list[str] | None = None) -> pd.DataFrame:
    """Run all (or a subset of) pipelines and return a ranked results DataFrame.

    Default set is read from config.yaml[pipelines][enabled] (A–F).
    Pass an explicit list or use --pipeline on the CLI to run G–J.
    """
    from src.utils.helpers import load_config
    if pipelines is None:
        cfg_enabled = load_config().get("pipelines", {}).get("enabled", list(_RUNNERS))
        names = [p for p in cfg_enabled if p in _RUNNERS]
    else:
        names = pipelines
    results = []
    for name in names:
        try:
            results.append(run_pipeline(name))
        except Exception as e:
            log.error("Pipeline %s failed: %s", name, e, exc_info=True)
            results.append({"pipeline": name, "error": str(e)})

    df = pd.DataFrame(results).sort_values("pr_auc_s1", ascending=False)

    _ARTIFACTS.mkdir(parents=True, exist_ok=True)
    df.to_csv(_ARTIFACTS / "results_all.csv", index=False)
    return df
