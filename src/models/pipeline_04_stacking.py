"""Pipeline 4: Stacking Ensemble (Level-2 Meta-Learning).

Level 1 base learners: LightGBM, XGBoost, CatBoost, RandomForest, ExtraTrees
Level 2 meta-learner: Logistic Regression (L2, C tuned on inner CV)

OOF generation follows strict K-fold to prevent leakage.
Meta-features = concatenated L1 probabilities (5 models × 3 classes = 15 features).
Expected time: ~1 hour.
"""
from __future__ import annotations

import numpy as np
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline as SklearnPipeline

from src.models.pipeline_utils import (
    ARTIFACTS, evaluate_proba, load_train_data,
    make_holdout, save_oof, save_result,
)
from src.utils.logger import get_logger

log = get_logger(__name__)
NAME = "P04_stacking"
CV_FOLDS = 5


def _make_base_learners() -> list[tuple[str, object]]:
    return [
        ("lgbm", lgb.LGBMClassifier(
            objective="multiclass", num_class=3, n_estimators=500,
            learning_rate=0.05, num_leaves=63, class_weight="balanced",
            n_jobs=-1, random_state=42, verbose=-1,
        )),
        ("xgb", xgb.XGBClassifier(
            objective="multi:softprob", num_class=3, n_estimators=500,
            learning_rate=0.05, max_depth=6, tree_method="hist",
            n_jobs=-1, random_state=42, verbosity=0,
        )),
        ("catboost", CatBoostClassifier(
            iterations=500, learning_rate=0.05, depth=6,
            auto_class_weights="Balanced", loss_function="MultiClass",
            random_seed=42, verbose=False, thread_count=-1,
        )),
        ("rf", RandomForestClassifier(
            n_estimators=500, max_depth=None, max_features="sqrt",
            class_weight="balanced", n_jobs=-1, random_state=42,
        )),
        ("et", ExtraTreesClassifier(
            n_estimators=500, max_depth=None, max_features="sqrt",
            class_weight="balanced", n_jobs=-1, random_state=42,
        )),
    ]


def run() -> dict:
    log.info("=" * 60)
    log.info("Pipeline 4: Stacking Ensemble")
    log.info("=" * 60)

    X, y = load_train_data()
    X_tv, X_hold, y_tv, y_hold = make_holdout(X, y)

    base_learners = _make_base_learners()
    n_base = len(base_learners)

    # ── Level 1: generate OOF meta-features ───────────────────────────────────
    oof_meta = np.zeros((len(X_tv), n_base * 3))
    test_meta = np.zeros((len(X_hold), n_base * 3))

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=42)

    for i, (base_name, base_clf) in enumerate(base_learners):
        log.info("Level 1 base learner: %s", base_name)
        col_start = i * 3
        test_fold_preds = []

        for fold, (tr, val) in enumerate(cv.split(X_tv, y_tv)):
            from sklearn.base import clone
            m = clone(base_clf)
            m.fit(X_tv.iloc[tr], y_tv[tr])
            oof_meta[val, col_start:col_start + 3] = m.predict_proba(X_tv.iloc[val])
            test_fold_preds.append(m.predict_proba(X_hold))
            log.info("  %s fold %d done", base_name, fold + 1)

        # Average test predictions across folds
        test_meta[:, col_start:col_start + 3] = np.mean(test_fold_preds, axis=0)

    # ── Level 2: meta-learner ─────────────────────────────────────────────────
    log.info("Training Level 2 meta-learner (LogisticRegressionCV)...")
    meta_learner = SklearnPipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegressionCV(
            Cs=[0.001, 0.01, 0.1, 1, 10], cv=5,
            penalty="l2", multi_class="multinomial",
            solver="lbfgs", max_iter=1000, n_jobs=-1, random_state=42,
        )),
    ])
    meta_learner.fit(oof_meta, y_tv)

    oof_final = meta_learner.predict_proba(oof_meta)
    cv_result = evaluate_proba(f"{NAME}_oof", y_tv, oof_final)
    save_result(cv_result)
    save_oof("p04", oof_final)

    # Holdout evaluation
    hold_final = meta_learner.predict_proba(test_meta)
    hold_result = evaluate_proba(f"{NAME}_holdout", y_hold, hold_final)
    save_result(hold_result)
    np.save(ARTIFACTS / "oof_p04_holdout.npy", hold_final)
    np.save(ARTIFACTS / "oof_p04_meta_tv.npy", oof_meta)    # L1 meta-features for P10

    return hold_result


if __name__ == "__main__":
    run()
