"""Orchestrates all feature groups into a single wide user-level DataFrame.

Usage:
    from src.features.build_features import build_feature_matrix

    X_train, y_train = build_feature_matrix("train")
    X_test, _        = build_feature_matrix("test")
"""
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.load_data import load_split
from src.data.preprocess import preprocess_all
from src.features.churn_features import (
    build_generation_features,
    build_properties_features,
    build_purchase_features,
    build_quiz_features,
    build_transaction_features,
)
from src.utils.helpers import load_config, normalize_country, processed_path
from src.utils.logger import get_logger

log = get_logger(__name__)

_LABEL_MAP = {"not_churned": 0, "vol_churn": 1, "invol_churn": 2}


# ── Cross-table features ───────────────────────────────────────────────────────

def _build_cross_table_features(
    gen_feat: pd.DataFrame,
    txn_feat: pd.DataFrame,
    pur_feat: pd.DataFrame,
    prop_feat: pd.DataFrame,
    quiz_feat: pd.DataFrame,
    props_raw: pd.DataFrame,
    obs_date: pd.Timestamp,
) -> pd.DataFrame:
    """Compute features that require joining multiple feature tables."""
    t = pd.Timestamp(obs_date)
    obs_date = t.tz_convert("UTC") if t.tzinfo is not None else t.tz_localize("UTC")
    all_users = prop_feat.index

    # Reindex all feature tables to the full user set so that Series comparisons
    # (which require identical indices) and arithmetic all work correctly.
    gen_feat = gen_feat.reindex(all_users)
    txn_feat = txn_feat.reindex(all_users)
    pur_feat = pur_feat.reindex(all_users)
    quiz_feat = quiz_feat.reindex(all_users)

    parts = []

    # X1: payment failure timing vs last generation activity
    # Positive = still generating when payment failed → INV
    # Negative = stopped generating before payment lapsed → VOL
    if "days_since_last_generation" in gen_feat.columns and "days_since_last_failed_transaction" in txn_feat.columns:
        x1 = (
            txn_feat["days_since_last_failed_transaction"] - gen_feat["days_since_last_generation"]
        ).rename("payment_failure_timing_vs_activity")
        parts.append(x1)

    # X2: was user actively generating within 7 days of last failed payment?
    if "days_since_last_generation" in gen_feat.columns and "days_since_last_failed_transaction" in txn_feat.columns:
        gap = gen_feat["days_since_last_generation"] - txn_feat["days_since_last_failed_transaction"]
        x2 = (gap.abs() <= 7).astype(int).rename("active_during_payment_failure")
        parts.append(x2)

    # X3: days from last purchase to last generation
    if "days_since_last_purchase" in pur_feat.columns and "days_since_last_generation" in gen_feat.columns:
        x3 = (pur_feat["days_since_last_purchase"] - gen_feat["days_since_last_generation"]).rename(
            "days_last_purchase_to_last_gen"
        )
        parts.append(x3)

    # X4: spend per generation (value perception)
    if "total_purchase_dollars" in pur_feat.columns and "total_generations" in gen_feat.columns:
        x4 = (pur_feat["total_purchase_dollars"] / (gen_feat["total_generations"] + 1)).rename("spend_per_generation")
        parts.append(x4)

    # X5/X6: plan credit utilisation
    if "plan_monthly_credits" in prop_feat.columns and "total_credits_consumed" in gen_feat.columns:
        tenure_months = (prop_feat["tenure_days"] / 30.0).clip(lower=0.5)
        plan_credit_budget = prop_feat["plan_monthly_credits"] * tenure_months
        x5 = (gen_feat["total_credits_consumed"] / plan_credit_budget.replace(0, np.nan)).fillna(0).rename(
            "plan_credit_utilization_pct"
        )
        x6 = (plan_credit_budget - gen_feat["total_credits_consumed"]).rename("plan_credit_surplus_deficit")
        parts.extend([x5, x6])

    # X7: time from subscription start to first failed transaction
    if "days_since_last_failed_transaction" in txn_feat.columns and "tenure_days" in prop_feat.columns:
        x7 = (prop_feat["tenure_days"] - txn_feat["days_since_last_failed_transaction"]).clip(lower=0).rename(
            "time_to_first_payment_issue_days"
        )
        parts.append(x7)

    # X8: generations per purchase (value extraction efficiency)
    if "total_generations" in gen_feat.columns and "n_purchases_total" in pur_feat.columns:
        x8 = (gen_feat["total_generations"] / (pur_feat["n_purchases_total"] + 1)).rename(
            "generation_to_purchase_ratio"
        )
        parts.append(x8)

    # X9: likely free-tier user (no purchases, no failed transactions)
    if "n_purchases_total" in pur_feat.columns and "n_failed_transactions" in txn_feat.columns:
        x9 = (
            (pur_feat["n_purchases_total"].fillna(0) == 0) &
            (txn_feat["n_failed_transactions"].fillna(0) == 0)
        ).astype(int).rename("is_likely_free_tier_user")
        parts.append(x9)

    # X10: generated before first purchase (trial-to-paid vs immediate buyer)
    if "_first_purchase" in pur_feat.columns and "days_to_first_generation" in gen_feat.columns:
        props_sub = props_raw.set_index("user_id")["subscription_start_date"]
        first_gen_abs = (
            props_sub + pd.to_timedelta(gen_feat["days_to_first_generation"], unit="D")
        ).reindex(all_users)
        x10 = (first_gen_abs < pur_feat["_first_purchase"]).astype(int).rename("generated_before_purchased")
        parts.append(x10)

    # X11: credit per dollar spent (value perception)
    if "total_credits_consumed" in gen_feat.columns and "total_purchase_dollars" in pur_feat.columns:
        x11 = (
            gen_feat["total_credits_consumed"] / (pur_feat["total_purchase_dollars"] + 1)
        ).rename("credit_per_dollar_spent")
        parts.append(x11)

    # X12: failed transactions before first successful one
    if "n_failed_without_matching_purchase" in txn_feat.columns:
        x12 = txn_feat["n_failed_without_matching_purchase"].rename("n_failed_txns_before_success")
        parts.append(x12)

    # X13: billing country matches profile country
    if "billing_address_country_norm" in props_raw.columns or True:
        # billing_address_country_norm is in transactions — attach from there if available
        pass  # computed in transaction features as billing_matches_profile_country
    if "billing_address_country" in props_raw.columns:
        pass  # properties doesn't have billing address; handled in transaction features
    # Derive from txn dominant billing country vs properties country_code
    # (best effort — transaction billing country modal value per user)

    # G26: feature expectation mismatch (wants video, uses mostly images)
    if "first_feature_video" in quiz_feat.columns and "video_generation_ratio" in gen_feat.columns:
        xg26 = (
            (quiz_feat["first_feature_video"] == 1) &
            (gen_feat["video_generation_ratio"] < 0.1)
        ).astype(int).rename("feature_expectation_mismatch")
        parts.append(xg26)

    if not parts:
        return pd.DataFrame(index=all_users)

    cross = pd.concat(parts, axis=1).reindex(all_users)
    cross.index.name = "user_id"
    return cross


def _build_composite_scores(feat: pd.DataFrame) -> pd.DataFrame:
    """Build interpretable composite scores from existing features."""
    scores = pd.DataFrame(index=feat.index)

    # CS1: Payment resilience score (INV risk — higher = more at risk)
    inv_cols = ["transaction_failure_rate", "uses_prepaid_card", "cvc_fail_rate", "n_3d_secure_friction"]
    available = [c for c in inv_cols if c in feat.columns]
    if available:
        normed = feat[available].copy()
        for c in available:
            mn, mx = normed[c].min(), normed[c].max()
            normed[c] = (normed[c] - mn) / (mx - mn + 1e-9)
        weights = {"transaction_failure_rate": 0.4, "uses_prepaid_card": 0.2,
                   "cvc_fail_rate": 0.2, "n_3d_secure_friction": 0.2}
        w = np.array([weights.get(c, 0.25) for c in available])
        w = w / w.sum()
        scores["payment_resilience_score"] = normed[available].values @ w

    # CS2: Engagement health score (VOL risk inverse — higher = healthier)
    eng_cols = ["completion_rate", "active_days_fraction", "gens_first_7_days", "engagement_trajectory_ratio"]
    available = [c for c in eng_cols if c in feat.columns]
    if available:
        normed = feat[available].copy()
        for c in available:
            mn, mx = normed[c].min(), normed[c].max()
            normed[c] = (normed[c] - mn) / (mx - mn + 1e-9)
        scores["engagement_health_score"] = normed[available].mean(axis=1)

    # CS3: Commitment score (quiz-based)
    comm_cols = ["team_size_ordinal", "experience_ordinal", "usage_plan_commercial", "role_commitment_score"]
    available = [c for c in comm_cols if c in feat.columns]
    if available:
        normed = feat[available].fillna(0).copy()
        for c in available:
            mn, mx = normed[c].min(), normed[c].max()
            normed[c] = (normed[c] - mn) / (mx - mn + 1e-9)
        scores["commitment_score"] = normed[available].mean(axis=1)

    # CS4–CS6: RFM bins (quartile 1-4)
    for rfm_src, rfm_name in [
        ("days_since_last_generation", "rfm_recency_bin"),
        ("generation_frequency_daily", "rfm_frequency_bin"),
        ("total_purchase_dollars", "rfm_monetary_bin"),
    ]:
        if rfm_src in feat.columns:
            col = feat[rfm_src].fillna(0)
            if rfm_src == "days_since_last_generation":
                col = -col  # invert: recent = high score
            scores[rfm_name] = pd.qcut(col, q=4, labels=[1, 2, 3, 4], duplicates="drop").astype(float)

    scores.index.name = "user_id"
    return scores


# ── Main entry point ───────────────────────────────────────────────────────────

def build_feature_matrix(
    split: str = "train",
    force_rebuild: bool = False,
) -> tuple[pd.DataFrame, pd.Series | None]:
    """Build or load cached feature matrix for a split.

    Returns:
        (X, y) where y is None for test split.
    """
    assert split in ("train", "test")
    cfg = load_config()
    obs_date = pd.Timestamp(cfg["observation_dates"][split], tz="UTC")

    out_dir = processed_path()
    out_dir.mkdir(parents=True, exist_ok=True)
    feat_path = out_dir / f"features_{split}.parquet"
    label_path = out_dir / f"labels_{split}.parquet"

    if feat_path.exists() and not force_rebuild:
        log.info("Loading cached features from %s", feat_path)
        X = pd.read_parquet(feat_path)
        y = pd.read_parquet(label_path).squeeze() if label_path.exists() else None
        return X, y

    # ── Load & preprocess ─────────────────────────────────────────────────────
    log.info("=== Building feature matrix for '%s' split ===", split)
    raw = load_split(split)
    tables = preprocess_all(raw)

    all_users = tables["users"]["user_id"].unique()
    all_users_idx = pd.Index(all_users, name="user_id")

    # ── Build per-table features ──────────────────────────────────────────────
    prop_feat = build_properties_features(tables["properties"], obs_date)
    gen_feat = build_generation_features(tables["generations"], tables["properties"], obs_date)
    pur_feat = build_purchase_features(tables["purchases"], obs_date)
    txn_feat = build_transaction_features(tables["transactions"], tables["purchases"], obs_date)
    quiz_feat = build_quiz_features(tables["quizzes"])

    # ── Cross-table features ──────────────────────────────────────────────────
    log.info("Building cross-table features ...")
    cross_feat = _build_cross_table_features(
        gen_feat=gen_feat,
        txn_feat=txn_feat,
        pur_feat=pur_feat,
        prop_feat=prop_feat,
        quiz_feat=quiz_feat,
        props_raw=tables["properties"],
        obs_date=obs_date,
    )

    # ── Merge all into wide table ─────────────────────────────────────────────
    log.info("Merging all feature groups ...")
    feat_groups = [prop_feat, gen_feat, pur_feat, txn_feat, quiz_feat, cross_feat]
    X = pd.concat(feat_groups, axis=1).reindex(all_users_idx)

    # Drop internal helper columns not meant as model features
    _drop = ["_first_purchase", "_last_purchase"]
    X = X.drop(columns=[c for c in _drop if c in X.columns])

    # ── Composite scores ──────────────────────────────────────────────────────
    log.info("Building composite scores ...")
    scores = _build_composite_scores(X)
    X = pd.concat([X, scores], axis=1)

    log.info("Feature matrix shape: %s", X.shape)

    # ── Labels ────────────────────────────────────────────────────────────────
    y = None
    if "churn_status" in tables["users"].columns:
        label_series = (
            tables["users"].set_index("user_id")["churn_status"]
            .map(_LABEL_MAP)
            .reindex(all_users_idx)
        )
        y = label_series
        y.to_frame().to_parquet(label_path)
        log.info("Label distribution:\n%s", y.value_counts().to_string())

    # ── Cache ─────────────────────────────────────────────────────────────────
    X.to_parquet(feat_path)
    log.info("Saved features to %s", feat_path)

    return X, y
