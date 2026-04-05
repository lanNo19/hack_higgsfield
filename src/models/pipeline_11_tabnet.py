"""Pipeline 11: TabNet + Diversity.

Train TabNet as an additional base learner. Add its OOF predictions to
the stacking ensemble from Pipeline 4.

Requires: pytorch-tabnet (pip install pytorch-tabnet).
If not installed, the pipeline logs a warning and exits gracefully.

Expected time: ~1 hour (TabNet is slow).
"""
from __future__ import annotations

import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from src.models.pipeline_utils import (
    ARTIFACTS, evaluate_proba, load_train_data,
    make_holdout, save_oof, save_result, load_oof,
)
from src.utils.logger import get_logger

log = get_logger(__name__)
NAME = "P11_tabnet"

try:
    from pytorch_tabnet.tab_model import TabNetClassifier  # type: ignore[import]
    TABNET_AVAILABLE = True
except ImportError:
    TABNET_AVAILABLE = False


def _train_tabnet(X_tr, y_tr, X_val, y_val) -> object:
    clf = TabNetClassifier(
        n_d=32, n_a=32, n_steps=5,
        gamma=1.5, n_independent=2, n_shared=2,
        momentum=0.02, epsilon=1e-15,
        optimizer_fn=__import__("torch").optim.Adam,
        optimizer_params={"lr": 2e-2},
        scheduler_params={"step_size": 50, "gamma": 0.9},
        scheduler_fn=__import__("torch").optim.lr_scheduler.StepLR,
        mask_type="sparsemax",
        verbose=0,
        seed=42,
    )
    clf.fit(
        X_tr.values, y_tr,
        eval_set=[(X_val.values, y_val)],
        eval_metric=["accuracy"],
        max_epochs=200,
        patience=20,
        batch_size=1024,
        virtual_batch_size=128,
    )
    return clf


def run() -> dict:
    log.info("=" * 60)
    log.info("Pipeline 11: TabNet + Diversity")
    log.info("=" * 60)

    if not TABNET_AVAILABLE:
        log.warning(
            "pytorch-tabnet not installed. Skipping Pipeline 11.\n"
            "Install with: pip install pytorch-tabnet"
        )
        result = {"pipeline": NAME, "error": "pytorch-tabnet not installed", "macro_f1": None}
        save_result(result)
        return result

    X, y = load_train_data()
    X_tv, X_hold, y_tv, y_hold = make_holdout(X, y)

    scaler = StandardScaler()
    X_tv_s = type(X_tv)(scaler.fit_transform(X_tv), columns=X_tv.columns, index=X_tv.index)
    X_hold_s = type(X_hold)(scaler.transform(X_hold), columns=X_hold.columns, index=X_hold.index)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_tabnet = np.zeros((len(X_tv), 3))

    for fold, (tr, val) in enumerate(cv.split(X_tv_s, y_tv)):
        clf = _train_tabnet(X_tv_s.iloc[tr], y_tv[tr], X_tv_s.iloc[val], y_tv[val])
        oof_tabnet[val] = clf.predict_proba(X_tv_s.iloc[val].values)
        log.info("Fold %d done", fold + 1)

    # TabNet standalone
    tn_result = evaluate_proba(f"{NAME}_standalone_oof", y_tv, oof_tabnet)
    save_result(tn_result)
    save_oof("p11_tabnet", oof_tabnet)

    # Blend TabNet OOF with P04 stacking OOF
    try:
        oof_stack = load_oof("p04")
        oof_blend = 0.7 * oof_stack + 0.3 * oof_tabnet
        blend_result = evaluate_proba(f"{NAME}_stack_blend_oof", y_tv, oof_blend)
        save_result(blend_result)
        save_oof("p11_blend", oof_blend)
    except FileNotFoundError:
        log.warning("P04 OOF not found — skipping blend.")

    # Holdout
    clf_final = _train_tabnet(X_tv_s, y_tv, X_hold_s, y_hold)
    hold_proba = clf_final.predict_proba(X_hold_s.values)
    hold_result = evaluate_proba(f"{NAME}_holdout", y_hold, hold_proba)
    save_result(hold_result)
    np.save(ARTIFACTS / "oof_p11_holdout.npy", hold_proba)

    return hold_result


if __name__ == "__main__":
    run()
