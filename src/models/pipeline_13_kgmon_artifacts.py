"""Pipeline 13: Synthetic Data Artifact Exploitation — KGMON Playbook.

Inspired by the 1st-place solution to Kaggle PS-S6E3 (churn prediction on
CTGAN-generated data). Our Higgsfield data is confirmed synthetic (year 1067
dates, Benford's Law violations, 33 anchor price points) — making these
techniques directly applicable.

Core idea: the generative model's fingerprints encode information about the
*original* data labels. Snap features, digit patterns, and Benford deviations
are features that tree models can exploit with high max_bin settings.

Phases:
  A. Infer anchor price grid from train+test frequency analysis
  B. Compute snap, digit, Benford, and row-level artifact features per user
  C. Augment existing feature matrix (+~50 artifact features)
  D. Train LightGBM + XGBoost + CatBoost with high max_bin / border_count
  E. Optionally stack with Pipeline 4 OOF for full ensemble diversity

Expected time: ~1.5 hours.
Priority: HIGH (elevated — artifact features feed downstream ensembles).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import optuna
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
from scipy.optimize import minimize
from sklearn.metrics import f1_score, log_loss
from sklearn.model_selection import StratifiedKFold

from src.models.pipeline_utils import (
    ARTIFACTS, CLASS_NAMES, evaluate_proba, load_train_data,
    make_holdout, save_oof, save_result, load_oof,
)
from src.utils.helpers import data_path, root_path
from src.utils.logger import get_logger

log = get_logger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)
NAME = "P13_kgmon"
N_TRIALS = 40
CV_FOLDS = 5
ANCHOR_MIN_FREQ = 40   # value must appear ≥ this many times to be an anchor


# ═══════════════════════════════════════════════════════════════════════════════
# A. Anchor Grid Inference
# ═══════════════════════════════════════════════════════════════════════════════

def _infer_anchor_grid(series: pd.Series, min_freq: int = ANCHOR_MIN_FREQ) -> np.ndarray:
    """Return sorted array of 'anchor' values that appear with high frequency."""
    freq = series.value_counts()
    anchors = freq[freq >= min_freq].index.to_numpy()
    return np.sort(anchors.astype(float))


def _snap(values: np.ndarray, anchors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Snap each value to its nearest anchor. Returns (snap_value, snap_dist)."""
    if len(anchors) == 0:
        return values.copy(), np.zeros_like(values)
    idx = np.searchsorted(anchors, values)
    idx = np.clip(idx, 0, len(anchors) - 1)
    left = np.clip(idx - 1, 0, len(anchors) - 1)
    dist_left = np.abs(values - anchors[left])
    dist_right = np.abs(values - anchors[idx])
    best = np.where(dist_left < dist_right, left, idx)
    return anchors[best], np.abs(values - anchors[best])


# ═══════════════════════════════════════════════════════════════════════════════
# B. Per-Transaction Feature Extraction
# ═══════════════════════════════════════════════════════════════════════════════

def _digit_features(x: np.ndarray, prefix: str) -> dict[str, np.ndarray]:
    """Extract decimal digit and rounding artifact features from a numeric array."""
    x = np.abs(x)
    frac = x - np.floor(x)
    d1 = np.floor(frac * 10).astype(int)
    d2 = (np.floor(frac * 100) % 10).astype(int)
    is_round = (frac < 0.005).astype(int)
    # quarter/half flags: |frac - 0.25|, |frac - 0.5|, |frac - 0.75| < 0.01
    is_quarter = (
        (np.abs(frac - 0.25) < 0.01) |
        (np.abs(frac - 0.50) < 0.01) |
        (np.abs(frac - 0.75) < 0.01)
    ).astype(int)
    mod10 = (np.floor(x) % 10).astype(int)
    return {
        f"{prefix}_d1":         d1,
        f"{prefix}_d2":         d2,
        f"{prefix}_is_round":   is_round,
        f"{prefix}_is_quarter": is_quarter,
        f"{prefix}_mod10":      mod10,
    }


def _benford_prob(x: np.ndarray) -> np.ndarray:
    """Benford's Law probability for the leading digit of each value."""
    out = np.zeros(len(x))
    pos = x > 0
    if pos.sum() == 0:
        return out
    digits = np.floor(
        np.abs(x[pos]) / 10 ** np.floor(np.log10(np.abs(x[pos]) + 1e-12))
    ).astype(int).clip(1, 9)
    out[pos] = np.log10(1 + 1.0 / digits)
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# C. Build Artifact Feature Matrix (User Level)
# ═══════════════════════════════════════════════════════════════════════════════

def build_artifact_features(
    user_ids: pd.Series,
    train_txn: pd.DataFrame,
    test_txn: pd.DataFrame,
    train_purch: pd.DataFrame,
    test_purch: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build a user-level artifact feature DataFrame.
    Joins back to user_ids so the index aligns with the main feature matrix.
    """
    # ── Anchor grids (fit on train+test combined) ──────────────────────────────
    all_amounts = pd.concat([
        train_txn["amount_in_usd"].dropna(),
        test_txn["amount_in_usd"].dropna(),
    ])
    all_purch = pd.concat([
        train_purch["purchase_amount_dollars"].dropna(),
        test_purch["purchase_amount_dollars"].dropna(),
    ])

    txn_anchors = _infer_anchor_grid(all_amounts)
    purch_anchors = _infer_anchor_grid(all_purch)
    log.info("Anchor grids: txn=%d prices, purch=%d prices", len(txn_anchors), len(purch_anchors))

    # ── Transaction artifact features ─────────────────────────────────────────
    txn = train_txn[["user_id", "amount_in_usd"]].dropna().copy()
    txn_amounts = txn["amount_in_usd"].values.astype(float)

    snap_val, snap_dist = _snap(txn_amounts, txn_anchors)
    txn["txn_snap_val"]  = snap_val
    txn["txn_snap_dist"] = snap_dist
    txn["txn_snap_diff"] = txn_amounts - snap_val
    txn["txn_benford"]   = _benford_prob(txn_amounts)

    for col, arr in _digit_features(txn_amounts, "txn_amt").items():
        txn[col] = arr

    # Snap frequency: how often does this snap value appear in train? (generator oversampling)
    snap_freq = txn["txn_snap_val"].value_counts(normalize=True)
    txn["txn_snap_freq"] = txn["txn_snap_val"].map(snap_freq).fillna(0)

    txn_agg = txn.groupby("user_id").agg(
        txn_mean_snap_dist=("txn_snap_dist", "mean"),
        txn_max_snap_dist=("txn_snap_dist", "max"),
        txn_mean_snap_diff=("txn_snap_diff", "mean"),
        txn_std_snap_diff=("txn_snap_diff", "std"),
        txn_mean_benford=("txn_benford", "mean"),
        txn_min_benford=("txn_benford", "min"),
        txn_n_round=("txn_amt_is_round", "sum"),
        txn_n_quarter=("txn_amt_is_quarter", "sum"),
        txn_pct_round=("txn_amt_is_round", "mean"),
        txn_pct_quarter=("txn_amt_is_quarter", "mean"),
        txn_snap_freq_mean=("txn_snap_freq", "mean"),
        txn_d1_mode=("txn_amt_d1", lambda x: x.mode().iloc[0] if len(x) > 0 else -1),
        txn_mod10_mode=("txn_amt_mod10", lambda x: x.mode().iloc[0] if len(x) > 0 else -1),
    ).reset_index()

    # ── Purchase artifact features ─────────────────────────────────────────────
    purch = train_purch[["user_id", "purchase_amount_dollars"]].dropna().copy()
    purch_amounts = purch["purchase_amount_dollars"].values.astype(float)

    psnap_val, psnap_dist = _snap(purch_amounts, purch_anchors)
    purch["purch_snap_val"]  = psnap_val
    purch["purch_snap_dist"] = psnap_dist
    purch["purch_snap_diff"] = purch_amounts - psnap_val
    purch["purch_benford"]   = _benford_prob(purch_amounts)

    for col, arr in _digit_features(purch_amounts, "purch_amt").items():
        purch[col] = arr

    purch_snap_freq = purch["purch_snap_val"].value_counts(normalize=True)
    purch["purch_snap_freq"] = purch["purch_snap_val"].map(purch_snap_freq).fillna(0)

    purch_agg = purch.groupby("user_id").agg(
        purch_mean_snap_dist=("purch_snap_dist", "mean"),
        purch_max_snap_dist=("purch_snap_dist", "max"),
        purch_mean_snap_diff=("purch_snap_diff", "mean"),
        purch_mean_benford=("purch_benford", "mean"),
        purch_n_round=("purch_amt_is_round", "sum"),
        purch_pct_round=("purch_amt_is_round", "mean"),
        purch_snap_freq_mean=("purch_snap_freq", "mean"),
        purch_d1_mode=("purch_amt_d1", lambda x: x.mode().iloc[0] if len(x) > 0 else -1),
    ).reset_index()

    # ── Merge to user list ─────────────────────────────────────────────────────
    feat = pd.DataFrame({"user_id": user_ids.values})
    feat = feat.merge(txn_agg, on="user_id", how="left")
    feat = feat.merge(purch_agg, on="user_id", how="left")
    feat = feat.drop(columns=["user_id"])
    feat = feat.fillna(0)

    log.info("Artifact features: %d columns", feat.shape[1])
    return feat


# ═══════════════════════════════════════════════════════════════════════════════
# D. Model Training with High Bin Count
# ═══════════════════════════════════════════════════════════════════════════════

def _tune_high_bin(X, y, model_type: str, n_trials: int = N_TRIALS) -> dict:
    log.info("Tuning %s (high-bin, %d trials)...", model_type, n_trials)

    def objective(trial: optuna.Trial) -> float:
        if model_type == "lgbm":
            params = dict(
                objective="multiclass", num_class=3,
                n_estimators=trial.suggest_int("n_estimators", 300, 1500),
                learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                num_leaves=trial.suggest_int("num_leaves", 31, 255),
                min_child_samples=trial.suggest_int("min_child_samples", 5, 80),
                subsample=trial.suggest_float("subsample", 0.5, 1.0), subsample_freq=1,
                colsample_bytree=trial.suggest_float("colsample_bytree", 0.3, 1.0),
                reg_alpha=trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
                reg_lambda=trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
                max_bin=8192,   # ← high bin count for digit features
                class_weight="balanced", n_jobs=-1, random_state=42, verbose=-1,
            )
            clf = lgb.LGBMClassifier(**params)

        elif model_type == "xgb":
            params = dict(
                objective="multi:softprob", num_class=3,
                n_estimators=trial.suggest_int("n_estimators", 300, 1500),
                learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                max_depth=trial.suggest_int("max_depth", 3, 9),
                min_child_weight=trial.suggest_int("min_child_weight", 1, 20),
                subsample=trial.suggest_float("subsample", 0.5, 1.0),
                colsample_bytree=trial.suggest_float("colsample_bytree", 0.3, 1.0),
                reg_alpha=trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
                reg_lambda=trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
                max_bin=16000,  # ← XGBoost high bin count
                tree_method="hist", n_jobs=-1, random_state=42, verbosity=0,
            )
            clf = xgb.XGBClassifier(**params)

        else:  # catboost
            params = dict(
                iterations=trial.suggest_int("iterations", 300, 1500),
                learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                depth=trial.suggest_int("depth", 4, 9),
                l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 1, 20, log=True),
                border_count=254,  # ← CatBoost max bin count
                random_strength=trial.suggest_float("random_strength", 0.1, 5.0),
                auto_class_weights="Balanced",
                loss_function="MultiClass",
                random_seed=42, verbose=False, thread_count=-1,
            )
            clf = CatBoostClassifier(**params)

        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        scores = []
        for tr, val in cv.split(X, y):
            clf_clone = type(clf)(**clf.get_params())
            clf_clone.fit(X.iloc[tr], y[tr])
            preds = np.argmax(clf_clone.predict_proba(X.iloc[val]), axis=1)
            scores.append(f1_score(y[val], preds, average="macro", zero_division=0))
        return float(np.mean(scores))

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    log.info("%s best macro_f1=%.4f", model_type, study.best_value)
    return study.best_params


def _oof_cv(build_fn, X, y) -> np.ndarray:
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=42)
    oof = np.zeros((len(X), 3))
    for fold, (tr, val) in enumerate(cv.split(X, y)):
        m = build_fn()
        m.fit(X.iloc[tr], y[tr])
        oof[val] = m.predict_proba(X.iloc[val])
        log.info("  fold %d done", fold + 1)
    return oof


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def run(n_trials: int = N_TRIALS) -> dict:
    log.info("=" * 60)
    log.info("Pipeline 13: KGMON Artifact Exploitation")
    log.info("=" * 60)

    # ── Load processed feature matrix ─────────────────────────────────────────
    X, y = load_train_data()
    X_tv, X_hold, y_tv, y_hold = make_holdout(X, y)

    # We need user_ids to join artifact features — reload raw to get them
    from src.utils.helpers import processed_path
    raw = pd.read_parquet(processed_path() / "features_train.parquet")
    if "user_id" not in raw.columns:
        log.warning("user_id not found in features_train.parquet — artifact join may be misaligned")
        user_ids_full = pd.Series(range(len(raw)))
    else:
        user_ids_full = raw["user_id"].reset_index(drop=True)

    # Split user_ids the same way (same random_state=42, holdout_size=0.15)
    from sklearn.model_selection import train_test_split
    uid_tv, uid_hold = train_test_split(
        user_ids_full, test_size=0.15, stratify=y, random_state=42
    )

    # ── Load raw CSVs for artifact engineering ─────────────────────────────────
    train_dir = data_path("train")
    test_dir  = data_path("test")

    def _safe_csv(path):
        if path.exists():
            return pd.read_csv(path)
        log.warning("Missing: %s", path)
        return pd.DataFrame()

    train_txn   = _safe_csv(train_dir / "train_users_transaction_attempts.csv")
    test_txn    = _safe_csv(test_dir  / "test_users_transaction_attempts.csv")
    train_purch = _safe_csv(train_dir / "train_users_purchases.csv")
    test_purch  = _safe_csv(test_dir  / "test_users_purchases.csv")

    # Verify required columns
    for df, col in [(train_txn, "amount_in_usd"), (train_purch, "purchase_amount_dollars")]:
        if col not in df.columns:
            log.warning("Column %s missing — artifact features will be zeros", col)

    # ── Build artifact features ────────────────────────────────────────────────
    log.info("Building artifact features (train set)...")
    art_tv   = build_artifact_features(uid_tv.reset_index(drop=True),
                                        train_txn, test_txn, train_purch, test_purch)
    log.info("Building artifact features (holdout set)...")
    art_hold = build_artifact_features(uid_hold.reset_index(drop=True),
                                        train_txn, test_txn, train_purch, test_purch)

    # ── Augment feature matrices ───────────────────────────────────────────────
    X_tv_aug   = pd.concat([X_tv.reset_index(drop=True),   art_tv],   axis=1)
    X_hold_aug = pd.concat([X_hold.reset_index(drop=True), art_hold], axis=1)

    log.info("Augmented feature matrix: %d → %d features", X_tv.shape[1], X_tv_aug.shape[1])

    # ── Train & evaluate three high-bin models ─────────────────────────────────
    model_specs = [
        ("lgbm_highbin", "lgbm",
         lambda p: lgb.LGBMClassifier(
             objective="multiclass", num_class=3, class_weight="balanced",
             max_bin=8192, n_jobs=-1, random_state=42, verbose=-1, **p)),
        ("xgb_highbin", "xgb",
         lambda p: xgb.XGBClassifier(
             objective="multi:softprob", num_class=3, tree_method="hist",
             max_bin=16000, n_jobs=-1, random_state=42, verbosity=0, **p)),
        ("cat_highbin", "catboost",
         lambda p: CatBoostClassifier(
             loss_function="MultiClass", border_count=254,
             auto_class_weights="Balanced", random_seed=42,
             verbose=False, thread_count=-1, **p)),
    ]

    all_oof_tv   = []
    all_hold_proba = []
    results = {}

    for model_name, tune_type, build_fn in model_specs:
        params = _tune_high_bin(X_tv_aug, y_tv, tune_type, n_trials)
        log.info("Generating OOF for %s...", model_name)
        oof = _oof_cv(lambda p=params, f=build_fn: f(p), X_tv_aug, y_tv)
        save_oof(f"p13_{model_name}", oof)

        cv_res = evaluate_proba(f"P13_{model_name}_oof", y_tv, oof)
        save_result(cv_res)

        m_final = build_fn(params)
        m_final.fit(X_tv_aug, y_tv)
        hold_p = m_final.predict_proba(X_hold_aug)
        hold_res = evaluate_proba(f"P13_{model_name}_holdout", y_hold, hold_p)
        save_result(hold_res)
        np.save(ARTIFACTS / f"oof_p13_{model_name}_holdout.npy", hold_p)

        all_oof_tv.append(oof)
        all_hold_proba.append(hold_p)
        results[model_name] = hold_res

    # ── Blend three high-bin models (equal weight) ─────────────────────────────
    oof_p13_blend   = np.mean(all_oof_tv, axis=0)
    hold_p13_blend  = np.mean(all_hold_proba, axis=0)

    blend_cv_res   = evaluate_proba("P13_blend_oof",     y_tv,   oof_p13_blend)
    blend_hold_res = evaluate_proba("P13_blend_holdout", y_hold, hold_p13_blend)
    save_result(blend_cv_res)
    save_result(blend_hold_res)
    save_oof("p13_blend", oof_p13_blend)
    np.save(ARTIFACTS / "oof_p13_blend_holdout.npy", hold_p13_blend)

    # ── Stack P13 blend into P04 stacking output ───────────────────────────────
    try:
        oof_p04   = load_oof("p04")
        hold_p04  = np.load(ARTIFACTS / "oof_p04_holdout.npy")

        # Optimize blend weight on trainval OOF
        def _loss(alpha):
            blended = alpha * oof_p04 + (1 - alpha) * oof_p13_blend
            return log_loss(y_tv, blended)

        from scipy.optimize import minimize_scalar
        res = minimize_scalar(_loss, bounds=(0, 1), method="bounded")
        alpha = float(res.x)
        log.info("P04+P13 blend alpha (P04 weight): %.4f", alpha)

        stacked_oof  = alpha * oof_p04  + (1 - alpha) * oof_p13_blend
        stacked_hold = alpha * hold_p04 + (1 - alpha) * hold_p13_blend

        stack_cv_res   = evaluate_proba("P13_plus_P04_oof",     y_tv,   stacked_oof,
                                        extra={"alpha_p04": round(alpha, 4)})
        stack_hold_res = evaluate_proba("P13_plus_P04_holdout", y_hold, stacked_hold,
                                        extra={"alpha_p04": round(alpha, 4)})
        save_result(stack_cv_res)
        save_result(stack_hold_res)
        save_oof("p13_plus_p04", stacked_oof)
        np.save(ARTIFACTS / "oof_p13_plus_p04_holdout.npy", stacked_hold)

        results["plus_p04"] = stack_hold_res
        log.info("P13+P04 stacked holdout macro_f1=%.4f", stack_hold_res["macro_f1"])

    except FileNotFoundError:
        log.info("P04 OOF not found — skipping P04+P13 stacking (run P04 first).")

    return blend_hold_res


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-trials", type=int, default=N_TRIALS)
    args = parser.parse_args()
    run(n_trials=args.n_trials)
