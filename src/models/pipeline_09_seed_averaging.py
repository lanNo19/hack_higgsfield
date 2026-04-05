"""Pipeline 9: Seed Averaging (Variance Reduction).

Train the best ensemble configuration (from P04 stacking) with 10 different
random seeds and average the predictions. Free variance reduction.

Expected improvement: 0.1-0.3% F1.
Expected time: ~10× Pipeline 4.
"""
from __future__ import annotations

import numpy as np
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.preprocessing import StandardScaler

from src.models.pipeline_utils import (
    ARTIFACTS, evaluate_proba, load_train_data,
    make_holdout, save_oof, save_result,
)
from src.utils.logger import get_logger

log = get_logger(__name__)
NAME = "P09_seed_averaging"
SEEDS = [42, 123, 256, 512, 999, 1337, 2024, 7777, 31415, 99999]
CV_FOLDS = 5


def _run_single_seed(X_tv, y_tv, X_hold, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Run full stacking pipeline with one seed. Returns (oof_tv, hold_proba)."""
    base_learners = [
        ("lgbm", lgb.LGBMClassifier(
            objective="multiclass", num_class=3, n_estimators=500,
            learning_rate=0.05, num_leaves=63, class_weight="balanced",
            n_jobs=-1, random_state=seed, verbose=-1,
        )),
        ("xgb", xgb.XGBClassifier(
            objective="multi:softprob", num_class=3, n_estimators=500,
            learning_rate=0.05, max_depth=6, tree_method="hist",
            n_jobs=-1, random_state=seed, verbosity=0,
        )),
        ("catboost", CatBoostClassifier(
            iterations=500, learning_rate=0.05, depth=6,
            auto_class_weights="Balanced", loss_function="MultiClass",
            random_seed=seed, verbose=False, thread_count=-1,
        )),
        ("rf", RandomForestClassifier(
            n_estimators=300, max_features="sqrt",
            class_weight="balanced", n_jobs=-1, random_state=seed,
        )),
        ("et", ExtraTreesClassifier(
            n_estimators=300, max_features="sqrt",
            class_weight="balanced", n_jobs=-1, random_state=seed,
        )),
    ]

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=seed)
    oof_meta = np.zeros((len(X_tv), len(base_learners) * 3))
    test_meta = np.zeros((len(X_hold), len(base_learners) * 3))

    for i, (_, clf) in enumerate(base_learners):
        col = i * 3
        test_preds = []
        for tr, val in cv.split(X_tv, y_tv):
            from sklearn.base import clone
            m = clone(clf)
            m.fit(X_tv.iloc[tr], y_tv[tr])
            oof_meta[val, col:col + 3] = m.predict_proba(X_tv.iloc[val])
            test_preds.append(m.predict_proba(X_hold))
        test_meta[:, col:col + 3] = np.mean(test_preds, axis=0)

    meta = SklearnPipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegressionCV(
            Cs=[0.01, 0.1, 1, 10], cv=3,
            penalty="l2", multi_class="multinomial",
            solver="lbfgs", max_iter=1000, n_jobs=-1, random_state=seed,
        )),
    ])
    meta.fit(oof_meta, y_tv)
    return meta.predict_proba(oof_meta), meta.predict_proba(test_meta)


def run(seeds: list[int] = SEEDS) -> dict:
    log.info("=" * 60)
    log.info("Pipeline 9: Seed Averaging (%d seeds)", len(seeds))
    log.info("=" * 60)

    X, y = load_train_data()
    X_tv, X_hold, y_tv, y_hold = make_holdout(X, y)

    all_oof_tv = []
    all_hold = []

    for i, seed in enumerate(seeds):
        log.info("Seed %d (%d/%d)...", seed, i + 1, len(seeds))
        oof_tv, hold_proba = _run_single_seed(X_tv, y_tv, X_hold, seed)
        all_oof_tv.append(oof_tv)
        all_hold.append(hold_proba)

    oof_avg = np.mean(all_oof_tv, axis=0)
    hold_avg = np.mean(all_hold, axis=0)

    cv_result = evaluate_proba(f"{NAME}_oof", y_tv, oof_avg,
                               extra={"n_seeds": len(seeds)})
    save_result(cv_result)
    save_oof("p09", oof_avg)

    hold_result = evaluate_proba(f"{NAME}_holdout", y_hold, hold_avg,
                                 extra={"n_seeds": len(seeds)})
    save_result(hold_result)
    np.save(ARTIFACTS / "oof_p09_holdout.npy", hold_avg)

    return hold_result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-seeds", type=int, default=len(SEEDS))
    args = parser.parse_args()
    run(seeds=SEEDS[: args.n_seeds])
