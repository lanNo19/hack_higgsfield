"""Pipeline 7: Imbalance Handling Ablation Study.

Tests multiple imbalance strategies on LightGBM (best single model from P02).
Strategies tested:
  1. raw — no rebalancing
  2. class_weight_balanced — LightGBM class_weight='balanced'
  3. smote — SMOTE oversampling (imblearn)
  4. smote_enn — SMOTE + ENN cleaning
  5. adasyn — Adaptive Synthetic Sampling
  6. focal_gamma1 — focal loss γ=1 (custom XGBoost objective)
  7. focal_gamma2 — focal loss γ=2
  8. focal_gamma5 — focal loss γ=5

Evaluation: Macro F1 on stratified 5-fold CV.
Expected time: ~1 hour.
"""
from __future__ import annotations

import numpy as np
import lightgbm as lgb
import xgboost as xgb
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.combine import SMOTEENN
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

from src.models.pipeline_utils import (
    evaluate_proba, load_train_data, make_holdout, save_result,
    LGBM_DEVICE, XGB_DEVICE,
)
from src.utils.logger import get_logger

log = get_logger(__name__)


# ── Focal loss custom objective for XGBoost ───────────────────────────────────

def _make_focal_objective(gamma: float, n_classes: int = 3):
    """Return a custom focal loss objective for XGBoost multiclass."""
    def focal_obj(labels, preds):
        # preds shape: (N * n_classes,) — XGBoost flattened
        preds = preds.reshape(len(labels), n_classes)
        # Softmax
        preds = preds - preds.max(axis=1, keepdims=True)
        exp_p = np.exp(preds)
        proba = exp_p / exp_p.sum(axis=1, keepdims=True)

        grad = np.zeros_like(preds)
        hess = np.zeros_like(preds)

        for k in range(n_classes):
            p_k = proba[:, k]
            y_k = (labels == k).astype(float)
            fl_weight = (1 - p_k) ** gamma
            grad[:, k] = fl_weight * (p_k - y_k)
            hess[:, k] = fl_weight * p_k * (1 - p_k) * (1 + gamma * np.log(p_k + 1e-9) * (p_k - y_k))

        return grad.flatten(), np.abs(hess).flatten() + 1e-6

    return focal_obj


def _lgbm_base_params(class_weight=None) -> dict:
    return dict(
        objective="multiclass",
        num_class=3,
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=63,
        min_child_samples=30,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        class_weight=class_weight,
        n_jobs=-1,
        random_state=42,
        verbose=-1,
        device=LGBM_DEVICE,
    )


def _cv_oof(build_fn, X, y, resampler=None, n_splits=5) -> tuple[np.ndarray, float]:
    """Run stratified CV with optional resampling. Returns OOF probas and mean macro F1."""
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    oof = np.zeros((len(X), 3))

    for tr, val in cv.split(X, y):
        X_tr, y_tr = X.iloc[tr], y[tr]

        if resampler is not None:
            X_tr_np, y_tr_res = resampler.fit_resample(X_tr, y_tr)
            X_tr_res = type(X_tr)(X_tr_np, columns=X_tr.columns)
        else:
            X_tr_res, y_tr_res = X_tr, y_tr

        m = build_fn()
        m.fit(X_tr_res, y_tr_res)
        oof[val] = m.predict_proba(X.iloc[val])

    macro_f1 = f1_score(y, np.argmax(oof, axis=1), average="macro", zero_division=0)
    return oof, macro_f1


def run() -> list[dict]:
    log.info("=" * 60)
    log.info("Pipeline 7: Imbalance Ablation")
    log.info("=" * 60)

    X, y = load_train_data()
    X_tv, X_hold, y_tv, y_hold = make_holdout(X, y)

    strategies = [
        # (name, build_fn, resampler)
        ("raw",
         lambda: lgb.LGBMClassifier(**_lgbm_base_params(class_weight=None)),
         None),
        ("class_weight_balanced",
         lambda: lgb.LGBMClassifier(**_lgbm_base_params(class_weight="balanced")),
         None),
        ("smote",
         lambda: lgb.LGBMClassifier(**_lgbm_base_params(class_weight=None)),
         SMOTE(random_state=42)),
        ("smote_enn",
         lambda: lgb.LGBMClassifier(**_lgbm_base_params(class_weight=None)),
         SMOTEENN(random_state=42)),
        ("adasyn",
         lambda: lgb.LGBMClassifier(**_lgbm_base_params(class_weight=None)),
         ADASYN(random_state=42)),
    ]

    # Focal loss via XGBoost custom objective
    for gamma in [1, 2, 5]:
        obj_fn = _make_focal_objective(gamma)

        def _make_focal_model(g=gamma):
            return xgb.XGBClassifier(
                num_class=3, n_estimators=500, learning_rate=0.05,
                max_depth=6, tree_method="hist", device=XGB_DEVICE, n_jobs=-1,
                random_state=42, verbosity=0,
                objective=_make_focal_objective(g),
                eval_metric="mlogloss",
            )

        strategies.append((f"focal_gamma{gamma}", _make_focal_model, None))

    results = []
    for strat_name, build_fn, resampler in strategies:
        log.info("Running strategy: %s", strat_name)
        try:
            oof, macro_f1 = _cv_oof(build_fn, X_tv, y_tv, resampler=resampler)
            result = evaluate_proba(f"P07_{strat_name}_oof", y_tv, oof,
                                    extra={"strategy": strat_name})
            save_result(result)
            results.append(result)
            log.info("  %s → macro_f1=%.4f", strat_name, macro_f1)
        except Exception as e:
            log.warning("Strategy %s failed: %s", strat_name, e)
            results.append({"pipeline": f"P07_{strat_name}", "error": str(e)})

    # Print comparison table
    log.info("\n── Ablation Results ──")
    for r in results:
        log.info("  %-30s  macro_f1=%s", r.get("pipeline", "?"), r.get("macro_f1", "ERR"))

    return results


if __name__ == "__main__":
    run()
