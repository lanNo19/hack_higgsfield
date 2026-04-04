"""Direct 3-class model builders for Pipelines K / L / M.

Labels: 0 = not_churned, 1 = vol_churn, 2 = invol_churn.
No two-stage cascade — single model predicts all three classes at once.
"""
from __future__ import annotations

import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier


def build_lgbm_multiclass(params: dict | None = None) -> lgb.LGBMClassifier:
    """LightGBM native 3-class classifier with balanced class weights."""
    default = dict(
        objective="multiclass",
        num_class=3,
        n_estimators=1000,
        learning_rate=0.05,
        num_leaves=95,
        min_child_samples=30,
        subsample=0.80,
        subsample_freq=1,
        colsample_bytree=0.70,
        reg_alpha=0.1,
        reg_lambda=1.0,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42,
        verbose=-1,
    )
    if params:
        default.update(params)
    return lgb.LGBMClassifier(**default)


def build_xgb_multiclass(params: dict | None = None) -> xgb.XGBClassifier:
    """XGBoost native 3-class classifier."""
    default = dict(
        objective="multi:softprob",
        num_class=3,
        n_estimators=700,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.80,
        colsample_bytree=0.75,
        reg_alpha=0.1,
        reg_lambda=2.0,
        eval_metric="mlogloss",
        n_jobs=-1,
        random_state=42,
        verbosity=0,
    )
    if params:
        default.update(params)
    return xgb.XGBClassifier(**default)


def build_catboost_multiclass(params: dict | None = None) -> CatBoostClassifier:
    """CatBoost native 3-class classifier with auto-balanced weights."""
    default = dict(
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
    if params:
        default.update(params)
    return CatBoostClassifier(**default)


def build_lgbm_s2_allfeat(params: dict | None = None) -> lgb.LGBMClassifier:
    """LightGBM Stage 2 using ALL features (vol vs invol, balanced data).

    Used by Pipeline N — unlike the existing T_FEATURES-only S2, this model
    can distinguish vol_churn via engagement signals too.
    """
    default = dict(
        n_estimators=800,
        learning_rate=0.05,
        num_leaves=63,
        min_child_samples=20,
        subsample=0.80,
        subsample_freq=1,
        colsample_bytree=0.70,
        reg_alpha=0.1,
        reg_lambda=1.0,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42,
        verbose=-1,
    )
    if params:
        default.update(params)
    return lgb.LGBMClassifier(**default)
