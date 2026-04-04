"""Stage 1 model builders: P(Churn) classifiers."""
from __future__ import annotations

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline as SKPipeline
from sklearn.preprocessing import StandardScaler

import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier


def build_lgbm_focal(params: dict | None = None) -> lgb.LGBMClassifier:
    """LightGBM with is_unbalance (focal loss applied externally via lgb.train if needed)."""
    default = dict(
        n_estimators=1000, learning_rate=0.05, num_leaves=95,
        min_child_samples=30, subsample=0.80, subsample_freq=1,
        colsample_bytree=0.70, reg_alpha=0.1, reg_lambda=1.0,
        is_unbalance=True, n_jobs=-1, random_state=42, verbose=-1,
    )
    if params:
        default.update(params)
    return lgb.LGBMClassifier(**default)


def build_lgbm_unbalanced(params: dict | None = None) -> lgb.LGBMClassifier:
    """LightGBM with is_unbalance and slightly wider trees."""
    default = dict(
        n_estimators=1000, learning_rate=0.05, num_leaves=63,
        min_child_samples=40, subsample=0.80, subsample_freq=1,
        colsample_bytree=0.70, is_unbalance=True,
        reg_alpha=0.1, reg_lambda=1.0,
        n_jobs=-1, random_state=42, verbose=-1,
    )
    if params:
        default.update(params)
    return lgb.LGBMClassifier(**default)


def build_catboost_s1(params: dict | None = None) -> CatBoostClassifier:
    """CatBoost Stage 1 with auto-balanced class weights."""
    default = dict(
        iterations=1000, learning_rate=0.05, depth=7, l2_leaf_reg=3.0,
        bagging_temperature=0.75, auto_class_weights="Balanced",
        eval_metric="PRAUC", random_seed=42, verbose=0,
        early_stopping_rounds=50,
    )
    if params:
        default.update(params)
    return CatBoostClassifier(**default)


def build_mlp_s1() -> SKPipeline:
    """MLP with standard scaling. Used as ensemble component in Pipelines E/F."""
    return SKPipeline([
        ("scaler", StandardScaler()),
        ("clf", MLPClassifier(
            hidden_layer_sizes=(256, 128, 64), activation="relu",
            solver="adam", alpha=1e-3, batch_size=1024,
            max_iter=100, random_state=42,
            early_stopping=True, validation_fraction=0.1,
        )),
    ])


def build_stacking_s1(scale_pos_weight: float = 1.0) -> CalibratedClassifierCV:
    """StackingClassifier (LGBM + XGB + CatBoost) with isotonic calibration.

    n_jobs=1 throughout: loky multiprocessing is broken on Python 3.13/Linux.
    Each base learner still uses internal threading (n_jobs=-1 per model).
    """
    base_lgbm = lgb.LGBMClassifier(
        n_estimators=1000, learning_rate=0.05, num_leaves=95,
        is_unbalance=True, subsample=0.80, colsample_bytree=0.70,
        n_jobs=-1, random_state=42, verbose=-1,
    )
    base_xgb = xgb.XGBClassifier(
        n_estimators=700, max_depth=5, learning_rate=0.05,
        scale_pos_weight=scale_pos_weight,
        eval_metric="aucpr", n_jobs=-1, random_state=42, verbosity=0,
    )
    base_cat = CatBoostClassifier(
        iterations=800, learning_rate=0.05, depth=7,
        auto_class_weights="Balanced", verbose=0, random_seed=42,
    )
    meta = LogisticRegression(C=1.0, max_iter=300, random_state=42)
    stacker = StackingClassifier(
        estimators=[("lgbm", base_lgbm), ("xgb", base_xgb), ("cat", base_cat)],
        final_estimator=meta,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        stack_method="predict_proba",
        passthrough=True,
        n_jobs=1,   # loky broken on Python 3.13/Linux
    )
    return CalibratedClassifierCV(stacker, method="isotonic", cv=3, n_jobs=1)
