"""
main.py — Churn prediction experiment runner (HackNU 2026 / Higgsfield)

Usage:
    python main.py                          # all pipelines, Optuna tuning
    python main.py --pipelines A B          # only A and B
    python main.py --no-optuna              # default params, faster
    python main.py --data-dir /path/data    # custom data directory
    python main.py --submission             # also produce hackathon submission CSV
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# ── make src importable whether run from root or src/ ────────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))

from src.data import (
    load_train_data,
    load_test_data,
    split_data,
    prepare_arrays,
    USER_ID_COL,
)
from src.features import (
    S1_FEATURES,
    T_FEATURES,
    CAT_FEATURES_S1,
    CAT_FEATURES_S2,
)
from src.pipeline_utils import (
    run_two_stage_cv,
    evaluate_pipeline,
    best_f1_threshold,
    predict_churn,
)
from src.pipelines import (
    # A
    build_lgbm_focal, build_xgb_s2,
    make_optuna_objective_A_s1, make_optuna_objective_A_s2,
    # B
    build_lgbm_unbalanced, build_logreg_s2,
    make_optuna_objective_B_s1, make_optuna_objective_B_s2,
    # C
    build_catboost_s1, build_catboost_s2,
    make_optuna_objective_C_s1, make_optuna_objective_C_s2,
    # D
    build_stacking_s1, build_voting_s2,
    make_optuna_objective_D_weights,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("main")


# ── Optuna trial budgets ──────────────────────────────────────────────────────
OPTUNA_TRIALS_S1 = 100
OPTUNA_TRIALS_S2 = 60
OPTUNA_TRIALS_D_WEIGHTS = 20


def run_optuna(objective_fn, n_trials: int, direction="maximize", study_name="study"):
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction=direction, study_name=study_name)
    study.optimize(objective_fn, n_trials=n_trials, show_progress_bar=True)
    logger.info(
        "Optuna [%s] best value=%.4f  params=%s",
        study_name, study.best_value, study.best_params,
    )
    return study.best_params


# ══════════════════════════════════════════════════════════════════════════════
# Per-pipeline training routines
# ══════════════════════════════════════════════════════════════════════════════

def train_pipeline_A(arrays: dict, use_optuna: bool) -> tuple:
    logger.info("=== Pipeline A: LightGBM (focal) + XGBoost ===")
    X_tr = arrays["X_train"]
    y_bin_tr = arrays["y_bin_train"].values
    y_vi_tr  = arrays["y_vi_train"].values

    s1_params, s2_params = {}, {}

    if use_optuna:
        logger.info("Tuning S1 (LightGBM) — %d trials", OPTUNA_TRIALS_S1)
        obj_s1 = make_optuna_objective_A_s1(X_tr, y_bin_tr, S1_FEATURES, T_FEATURES, y_vi_tr)
        s1_params = run_optuna(obj_s1, OPTUNA_TRIALS_S1, study_name="A_s1")

        logger.info("Tuning S2 (XGBoost) — %d trials", OPTUNA_TRIALS_S2)
        obj_s2 = make_optuna_objective_A_s2(X_tr, y_vi_tr, T_FEATURES)
        s2_params = run_optuna(obj_s2, OPTUNA_TRIALS_S2, study_name="A_s2")

    oof_s1, oof_s2 = run_two_stage_cv(
        build_s1=lambda: build_lgbm_focal(s1_params),
        build_s2=lambda: build_xgb_s2(y_vi_tr, s2_params),
        X=X_tr, y_binary=y_bin_tr, y_vol_inv=y_vi_tr,
        s1_feature_cols=S1_FEATURES, s2_feature_cols=T_FEATURES,
    )
    metrics = evaluate_pipeline(y_bin_tr, oof_s1, y_vi_tr, oof_s2, pipeline_name="A")

    # Fit final models on full train set
    s1_final = build_lgbm_focal(s1_params)
    s1_final.fit(
        X_tr[[c for c in S1_FEATURES if c in X_tr.columns]], y_bin_tr
    )
    s2_final = build_xgb_s2(y_vi_tr, s2_params)
    churn_mask = y_bin_tr == 1
    valid_vi   = churn_mask & ~np.isnan(y_vi_tr)
    s2_final.fit(
        X_tr[[c for c in T_FEATURES if c in X_tr.columns]][valid_vi],
        y_vi_tr[valid_vi].astype(int),
    )
    return s1_final, s2_final, metrics, metrics["best_threshold_s1"]


def train_pipeline_B(arrays: dict, use_optuna: bool) -> tuple:
    logger.info("=== Pipeline B: LightGBM (unbalanced) + LogReg ===")
    X_tr = arrays["X_train"]
    y_bin_tr = arrays["y_bin_train"].values
    y_vi_tr  = arrays["y_vi_train"].values

    s1_params, s2_params = {}, {}

    if use_optuna:
        logger.info("Tuning S1 (LightGBM is_unbalance) — %d trials", OPTUNA_TRIALS_S1)
        obj_s1 = make_optuna_objective_B_s1(X_tr, y_bin_tr, S1_FEATURES)
        s1_params = run_optuna(obj_s1, OPTUNA_TRIALS_S1, study_name="B_s1")

        logger.info("Tuning S2 (LogReg) — %d trials", OPTUNA_TRIALS_S2)
        obj_s2 = make_optuna_objective_B_s2(X_tr, y_vi_tr, T_FEATURES)
        s2_params = run_optuna(obj_s2, OPTUNA_TRIALS_S2, study_name="B_s2")

    oof_s1, oof_s2 = run_two_stage_cv(
        build_s1=lambda: build_lgbm_unbalanced(s1_params),
        build_s2=lambda: build_logreg_s2(s2_params),
        X=X_tr, y_binary=y_bin_tr, y_vol_inv=y_vi_tr,
        s1_feature_cols=S1_FEATURES, s2_feature_cols=T_FEATURES,
    )
    metrics = evaluate_pipeline(y_bin_tr, oof_s1, y_vi_tr, oof_s2, pipeline_name="B")

    s1_final = build_lgbm_unbalanced(s1_params)
    s1_final.fit(
        X_tr[[c for c in S1_FEATURES if c in X_tr.columns]], y_bin_tr
    )
    s2_final = build_logreg_s2(s2_params)
    churn_mask = y_bin_tr == 1
    valid_vi   = churn_mask & ~np.isnan(y_vi_tr)
    s2_final.fit(
        X_tr[[c for c in T_FEATURES if c in X_tr.columns]][valid_vi],
        y_vi_tr[valid_vi].astype(int),
    )
    return s1_final, s2_final, metrics, metrics["best_threshold_s1"]


def train_pipeline_C(arrays: dict, use_optuna: bool) -> tuple:
    logger.info("=== Pipeline C: CatBoost + CatBoost ===")
    X_tr = arrays["X_train"]
    y_bin_tr = arrays["y_bin_train"].values
    y_vi_tr  = arrays["y_vi_train"].values

    # Only use cat features that actually exist in data
    cat_s1 = [c for c in CAT_FEATURES_S1 if c in X_tr.columns]
    cat_s2 = [c for c in CAT_FEATURES_S2 if c in X_tr.columns]

    s1_params, s2_params = {}, {}

    if use_optuna:
        logger.info("Tuning S1 (CatBoost) — %d trials", OPTUNA_TRIALS_S1)
        obj_s1 = make_optuna_objective_C_s1(X_tr, y_bin_tr, S1_FEATURES, cat_s1)
        s1_params = run_optuna(obj_s1, OPTUNA_TRIALS_S1, study_name="C_s1")

        logger.info("Tuning S2 (CatBoost) — %d trials", OPTUNA_TRIALS_S2)
        obj_s2 = make_optuna_objective_C_s2(X_tr, y_vi_tr, T_FEATURES, cat_s2)
        s2_params = run_optuna(obj_s2, OPTUNA_TRIALS_S2, study_name="C_s2")

    oof_s1, oof_s2 = run_two_stage_cv(
        build_s1=lambda: build_catboost_s1(cat_s1, s1_params),
        build_s2=lambda: build_catboost_s2(y_vi_tr, cat_s2, s2_params),
        X=X_tr, y_binary=y_bin_tr, y_vol_inv=y_vi_tr,
        s1_feature_cols=S1_FEATURES, s2_feature_cols=T_FEATURES,
    )
    metrics = evaluate_pipeline(y_bin_tr, oof_s1, y_vi_tr, oof_s2, pipeline_name="C")

    s1_final = build_catboost_s1(cat_s1, s1_params)
    s1_final.fit(
        X_tr[[c for c in S1_FEATURES if c in X_tr.columns]], y_bin_tr
    )
    s2_final = build_catboost_s2(y_vi_tr, cat_s2, s2_params)
    churn_mask = y_bin_tr == 1
    valid_vi   = churn_mask & ~np.isnan(y_vi_tr)
    s2_final.fit(
        X_tr[[c for c in T_FEATURES if c in X_tr.columns]][valid_vi],
        y_vi_tr[valid_vi].astype(int),
    )
    return s1_final, s2_final, metrics, metrics["best_threshold_s1"]


def train_pipeline_D(arrays: dict, use_optuna: bool) -> tuple:
    logger.info("=== Pipeline D: Stacking Ensemble ===")
    X_tr = arrays["X_train"]
    y_bin_tr = arrays["y_bin_train"].values
    y_vi_tr  = arrays["y_vi_train"].values

    neg = (y_bin_tr == 0).sum()
    pos = (y_bin_tr == 1).sum()
    spw = neg / pos if pos > 0 else 1.0

    voting_params = {}
    if use_optuna:
        logger.info("Tuning S2 voting weights (Pipeline D) — %d trials", OPTUNA_TRIALS_D_WEIGHTS)
        obj_w = make_optuna_objective_D_weights(X_tr, y_vi_tr, T_FEATURES)
        best_w = run_optuna(obj_w, OPTUNA_TRIALS_D_WEIGHTS, study_name="D_weights")
        voting_params["weights"] = [best_w["w_xgb"], best_w["w_cat"], best_w["w_lr"]]

    oof_s1, oof_s2 = run_two_stage_cv(
        build_s1=lambda: build_stacking_s1(spw),
        build_s2=lambda: build_voting_s2(y_vi_tr, voting_params),
        X=X_tr, y_binary=y_bin_tr, y_vol_inv=y_vi_tr,
        s1_feature_cols=S1_FEATURES, s2_feature_cols=T_FEATURES,
    )
    metrics = evaluate_pipeline(y_bin_tr, oof_s1, y_vi_tr, oof_s2, pipeline_name="D")

    s1_final = build_stacking_s1(spw)
    s1_cols  = [c for c in S1_FEATURES if c in X_tr.columns]
    s1_final.fit(X_tr[s1_cols], y_bin_tr)

    s2_final = build_voting_s2(y_vi_tr, voting_params)
    churn_mask = y_bin_tr == 1
    valid_vi   = churn_mask & ~np.isnan(y_vi_tr)
    s2_final.fit(
        X_tr[[c for c in T_FEATURES if c in X_tr.columns]][valid_vi],
        y_vi_tr[valid_vi].astype(int),
    )
    return s1_final, s2_final, metrics, metrics["best_threshold_s1"]


# ══════════════════════════════════════════════════════════════════════════════
# Val / test evaluation helper
# ══════════════════════════════════════════════════════════════════════════════

def eval_on_split(
    split_df: pd.DataFrame,
    s1_model,
    s2_model,
    threshold: float,
    split_name: str,
) -> pd.DataFrame:
    from sklearn.metrics import average_precision_score, f1_score

    preds = predict_churn(
        split_df, s1_model, s2_model,
        s1_feature_cols=S1_FEATURES,
        s2_feature_cols=T_FEATURES,
        threshold_s1=threshold,
    )
    # Re-attach true labels for metrics
    preds = preds.merge(
        split_df[["user_id", "label"]], on="user_id", how="left"
    )

    from src.data import BINARY_MAP, VOL_INV_MAP
    y_true_bin = preds["label"].map(BINARY_MAP).fillna(0).astype(int)
    y_pred_bin = (preds["final_label"] != "not_churned").astype(int)

    pr_auc = average_precision_score(
        y_true_bin, preds["churn_probability"].fillna(0)
    )
    f1 = f1_score(y_true_bin, y_pred_bin, zero_division=0)
    logger.info(
        "[%s] PR-AUC=%.4f  F1=%.4f", split_name.upper(), pr_auc, f1
    )
    return preds


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

PIPELINE_TRAINERS = {
    "A": train_pipeline_A,
    "B": train_pipeline_B,
    "C": train_pipeline_C,
    "D": train_pipeline_D,
}


def main():
    parser = argparse.ArgumentParser(description="Higgsfield churn prediction")
    parser.add_argument("--pipelines", nargs="+", choices=["A", "B", "C", "D"],
                        default=["A", "B", "C", "D"])
    parser.add_argument("--no-optuna", action="store_true",
                        help="Skip Optuna and use default hyperparameters")
    parser.add_argument("--data-dir", default="data", help="Directory with parquet files")
    parser.add_argument("--output-dir", default="reports", help="Where to save results")
    parser.add_argument("--submission", action="store_true",
                        help="Run inference on features_test.parquet and save submission CSV")
    args = parser.parse_args()

    use_optuna = not args.no_optuna
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load & split data ─────────────────────────────────────────────────────
    logger.info("Loading training data from '%s'", args.data_dir)
    df = load_train_data(args.data_dir)
    train_df, val_df, test_df = split_data(df, val_size=0.10, test_size=0.10)

    # Prepare arrays once (all features combined; pipelines pick their subsets)
    all_feature_cols = list(
        dict.fromkeys(S1_FEATURES + T_FEATURES)
    )
    arrays = prepare_arrays(train_df, val_df, test_df, all_feature_cols)

    # Keep full DataFrames for gate + inference
    arrays["_train_df"] = train_df
    arrays["_val_df"]   = val_df
    arrays["_test_df"]  = test_df

    # ── Run pipelines ─────────────────────────────────────────────────────────
    all_results = []
    best_pipeline = None
    best_s1 = None
    best_s2 = None
    best_thr = 0.30
    best_pr_auc = -1.0

    for name in args.pipelines:
        trainer = PIPELINE_TRAINERS[name]
        try:
            s1, s2, metrics, thr = trainer(arrays, use_optuna)
            all_results.append({**metrics, "timestamp": datetime.now().isoformat()})

            # Evaluate on val split
            eval_on_split(val_df, s1, s2, thr, f"{name}-val")

            if metrics["pr_auc_s1"] > best_pr_auc:
                best_pr_auc  = metrics["pr_auc_s1"]
                best_pipeline = name
                best_s1, best_s2, best_thr = s1, s2, thr

        except Exception as exc:
            logger.error("Pipeline %s failed: %s", name, exc, exc_info=True)

    # ── Final test-set evaluation ─────────────────────────────────────────────
    if best_s1 is not None:
        logger.info("Best pipeline: %s (S1 PR-AUC=%.4f)", best_pipeline, best_pr_auc)
        eval_on_split(test_df, best_s1, best_s2, best_thr, f"{best_pipeline}-test")

    # ── Save experiment results ───────────────────────────────────────────────
    results_df = pd.DataFrame(all_results).sort_values("pr_auc_s1", ascending=False)
    results_path = out_dir / f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    results_df.to_csv(results_path, index=False)
    logger.info("Results saved → %s", results_path)
    print("\n=== Final Rankings ===")
    print(results_df.to_string(index=False))

    # ── Hackathon submission ──────────────────────────────────────────────────
    if args.submission and best_s1 is not None:
        logger.info("Generating submission for features_test.parquet …")
        test_holdout = load_test_data(args.data_dir)
        submission = predict_churn(
            test_holdout, best_s1, best_s2,
            s1_feature_cols=S1_FEATURES,
            s2_feature_cols=T_FEATURES,
            threshold_s1=best_thr,
        )
        sub_path = out_dir / "submission.csv"
        submission[["user_id", "final_label"]].to_csv(sub_path, index=False)
        logger.info("Submission saved → %s  (%d rows)", sub_path, len(submission))

    return results_df


if __name__ == "__main__":
    main()
