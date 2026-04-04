"""Stage 2 model builders: P(Involuntary | Churn) classifiers."""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import VotingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline as SKPipeline
from sklearn.preprocessing import StandardScaler

import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier


def build_xgb_s2(y_volInv: np.ndarray, params: dict | None = None) -> xgb.XGBClassifier:
    """XGBoost Stage 2 with scale_pos_weight derived from churned-subset class ratio."""
    neg = (y_volInv == 0).sum()
    pos = (y_volInv == 1).sum()
    spw = neg / max(pos, 1)
    default = dict(
        n_estimators=700, max_depth=5, learning_rate=0.05,
        scale_pos_weight=spw, min_child_weight=10, gamma=0.1,
        subsample=0.80, colsample_bytree=0.75,
        reg_alpha=0.1, reg_lambda=2.0,
        eval_metric="aucpr", random_state=42, n_jobs=-1, verbosity=0,
    )
    if params:
        default.update(params)
    return xgb.XGBClassifier(**default)


def build_logreg_s2(params: dict | None = None) -> SKPipeline:
    """Elastic-net Logistic Regression Stage 2. Coefficients are PM-readable."""
    default_lr = dict(
        C=1.0, penalty="elasticnet", solver="saga",
        l1_ratio=0.5, class_weight="balanced",
        max_iter=500, random_state=42,
    )
    if params:
        default_lr.update(params)
    return SKPipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
        ("clf",     LogisticRegression(**default_lr)),
    ])


def build_catboost_s2(y_volInv: np.ndarray, params: dict | None = None) -> CatBoostClassifier:
    """CatBoost Stage 2 with manually computed class weights."""
    neg = (y_volInv == 0).sum()
    pos = (y_volInv == 1).sum()
    default = dict(
        iterations=600, learning_rate=0.06, depth=6, l2_leaf_reg=5.0,
        class_weights={0: 1.0, 1: float(neg / max(pos, 1))},
        eval_metric="PRAUC", random_seed=42, verbose=0,
        early_stopping_rounds=40,
    )
    if params:
        default.update(params)
    return CatBoostClassifier(**default)


def build_lgbm_s2(params: dict | None = None) -> lgb.LGBMClassifier:
    """LightGBM Stage 2 for Pipelines E/F (T-features only)."""
    default = dict(
        n_estimators=700, learning_rate=0.05, num_leaves=63,
        is_unbalance=True, n_jobs=-1, random_state=42, verbose=-1,
    )
    if params:
        default.update(params)
    return lgb.LGBMClassifier(**default)


def build_voting_s2(y_volInv: np.ndarray) -> VotingClassifier:
    """Soft-voting ensemble Stage 2: XGB + CatBoost + LogReg."""
    spw = float((y_volInv == 0).sum() / max((y_volInv == 1).sum(), 1))
    xgb_s2 = xgb.XGBClassifier(
        n_estimators=700, max_depth=5, learning_rate=0.05,
        scale_pos_weight=spw, eval_metric="aucpr",
        n_jobs=-1, random_state=42, verbosity=0,
    )
    cat_s2 = CatBoostClassifier(
        iterations=600, depth=6, learning_rate=0.06,
        class_weights={0: 1, 1: spw}, verbose=0, random_seed=42,
    )
    lr_s2 = SKPipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
        ("clf", LogisticRegression(
            C=0.1, penalty="elasticnet", solver="saga",
            l1_ratio=0.5, class_weight="balanced", max_iter=500,
        )),
    ])
    return VotingClassifier(
        estimators=[("xgb", xgb_s2), ("cat", cat_s2), ("lr", lr_s2)],
        voting="soft", weights=[2, 2, 1], n_jobs=1,  # loky broken on Python 3.13/Linux
    )
