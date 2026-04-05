"""
Feature Engineering for Higgsfield AI Churn Prediction (HackNU 2026)
====================================================================

Three-class problem: not_churned | vol_churn | invol_churn

Design rationale (sourced from research):
- Voluntary churn signals: engagement decay, value perception gap, usage decline
- Involuntary churn signals: payment failures, card issues, billing friction
- Key insight from WSDM-KKBox 1st place: trend/temporal features >> snapshot features
- Key insight from SaaS literature: derived ratios & week-over-week deltas beat raw counts

Feature groups:
  1. SUBSCRIPTION / DEMOGRAPHICS  (properties table)
  2. PAYMENT BEHAVIOR             (purchases table)
  3. BILLING RISK / FRICTION      (transaction_attempts table)
  4. GENERATION ENGAGEMENT        (generations table)
  5. ONBOARDING / QUIZ            (quizzes table)
  6. CROSS-TABLE INTERACTIONS     (derived)

Usage:
    python -m src.features.feature_engineering --mode train
    python -m src.features.feature_engineering --mode test
    # or with explicit paths:
    python -m src.features.feature_engineering --mode train --data-dir ./data/raw/train --out ./data/processed/features_train.parquet

Expects file naming convention:
    data/raw/train/train_users.csv, train_users_properties.csv, ...
    data/raw/test/test_users.csv,   test_users_properties.csv,  ...

For generations: train_users_generations.csv / test_users_generations.csv
If a table is missing, those features are skipped gracefully.
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def safe_load(path: Path) -> pd.DataFrame | None:
    """Load CSV if exists, else return None."""
    if path.exists():
        return pd.read_csv(path)
    print(f"  [SKIP] {path.name} not found")
    return None


def parse_dt(series: pd.Series) -> pd.Series:
    """Parse datetime, coerce errors."""
    return pd.to_datetime(series, errors="coerce", utc=True)


def days_between(a: pd.Series, b: pd.Series) -> pd.Series:
    """Return (b - a) in fractional days."""
    return (b - a).dt.total_seconds() / 86400


# =====================================================================
# 1. SUBSCRIPTION & DEMOGRAPHIC FEATURES
# =====================================================================

def build_subscription_features(props: pd.DataFrame, ref_date: pd.Timestamp) -> pd.DataFrame:
    """
    From: properties table (user_id, subscription_start_date, subscription_plan, country_code)

    Features:
    - sub_tenure_days: days since subscription start → longer tenure = lower vol churn risk
    - plan_tier: ordinal encoding of plan (Creator < Basic < Pro < Ultimate)
    - is_top_tier_plan: binary flag for Ultimate plan
    - country_code_encoded: target-encoded or frequency-encoded country
    - is_high_churn_region: flag for countries with historically higher churn
    """
    df = props[["user_id"]].copy()

    # Tenure
    sub_start = parse_dt(props["subscription_start_date"])
    df["sub_tenure_days"] = days_between(sub_start, ref_date)

    # Day-of-week and hour of subscription (proxy for user type)
    df["sub_dow"] = sub_start.dt.dayofweek  # 0=Mon
    df["sub_hour"] = sub_start.dt.hour
    df["sub_is_weekend"] = (df["sub_dow"] >= 5).astype(int)

    # Plan tier (ordinal) — higher = more invested = less likely vol_churn
    plan_order = {
        "Higgsfield Creator": 0,
        "Higgsfield Basic": 1,
        "Higgsfield Pro": 2,
        "Higgsfield Ultimate": 3,
    }
    df["plan_tier"] = props["subscription_plan"].map(plan_order).fillna(-1).astype(int)
    df["is_top_tier_plan"] = (df["plan_tier"] == 3).astype(int)
    df["is_bottom_tier_plan"] = (df["plan_tier"] <= 0).astype(int)

    # Country frequency encoding (rare country → higher risk of payment issues)
    country_freq = props["country_code"].value_counts(normalize=True)
    df["country_freq"] = props["country_code"].map(country_freq).fillna(0)
    df["is_rare_country"] = (df["country_freq"] < 0.005).astype(int)

    # One-hot top countries (US, IN, DE, GB, KR, FR) — rest as "other"
    top_countries = ["US", "IN", "DE", "GB", "KR", "FR"]
    for c in top_countries:
        df[f"country_is_{c}"] = (props["country_code"] == c).astype(int)

    return df


# =====================================================================
# 2. PAYMENT / PURCHASE BEHAVIOR FEATURES
# =====================================================================

def build_purchase_features(purchases: pd.DataFrame, ref_date: pd.Timestamp) -> pd.DataFrame:
    """
    From: purchases table (user_id, transaction_id, purchase_time, purchase_type, purchase_amount_dollars)

    Rationale: payment patterns are the strongest signal for both churn types.
    - For involuntary: erratic amounts, low spending might correlate with payment issues
    - For voluntary: lack of upsells, no credit purchases = low engagement

    Features:
    - total_purchases, total_spend, avg_purchase_amount
    - purchase_type counts: n_sub_create, n_sub_update, n_credits, n_upsell, n_gift
    - has_credits_purchase: bought extra credits → engaged user
    - has_upsell: accepted upsell → high value perception
    - recency: days since last purchase
    - purchase_frequency: purchases per day of tenure
    - spend_variability: std / mean of amounts (CV)
    - time_between_purchases: mean/std inter-purchase interval
    - first_purchase_to_ref: time from first purchase
    - max_single_purchase: largest single purchase
    """
    purch = purchases.copy()
    purch["purchase_time"] = parse_dt(purch["purchase_time"])

    # ---- Aggregate per user ----
    agg = purch.groupby("user_id").agg(
        total_purchases=("transaction_id", "count"),
        total_spend=("purchase_amount_dollars", "sum"),
        avg_purchase_amount=("purchase_amount_dollars", "mean"),
        std_purchase_amount=("purchase_amount_dollars", "std"),
        max_purchase_amount=("purchase_amount_dollars", "max"),
        min_purchase_amount=("purchase_amount_dollars", "min"),
        first_purchase_time=("purchase_time", "min"),
        last_purchase_time=("purchase_time", "max"),
    ).reset_index()

    # Coefficient of variation of purchase amounts
    agg["spend_cv"] = agg["std_purchase_amount"] / (agg["avg_purchase_amount"] + 1e-9)
    agg["spend_range"] = agg["max_purchase_amount"] - agg["min_purchase_amount"]

    # Recency
    agg["days_since_last_purchase"] = days_between(agg["last_purchase_time"], ref_date)
    agg["days_since_first_purchase"] = days_between(agg["first_purchase_time"], ref_date)

    # Purchase frequency (purchases per day of observation window)
    purchase_window = days_between(agg["first_purchase_time"], agg["last_purchase_time"]).clip(lower=1)
    agg["purchase_frequency"] = agg["total_purchases"] / purchase_window

    # ---- Purchase type pivots ----
    type_counts = purch.groupby(["user_id", "purchase_type"]).size().unstack(fill_value=0)
    type_counts.columns = [f"n_purch_{c.lower().replace(' ', '_')}" for c in type_counts.columns]
    type_counts = type_counts.reset_index()

    agg = agg.merge(type_counts, on="user_id", how="left")

    # Binary flags
    credit_col = "n_purch_credits_package"
    upsell_col = "n_purch_upsell"
    gift_col = "n_purch_gift"

    if credit_col in agg.columns:
        agg["has_credits_purchase"] = (agg[credit_col] > 0).astype(int)
    else:
        agg["has_credits_purchase"] = 0

    if upsell_col in agg.columns:
        agg["has_upsell"] = (agg[upsell_col] > 0).astype(int)
    else:
        agg["has_upsell"] = 0

    if gift_col in agg.columns:
        agg["has_gift"] = (agg[gift_col] > 0).astype(int)
    else:
        agg["has_gift"] = 0

    # ---- Inter-purchase intervals ----
    intervals = (
        purch.sort_values("purchase_time")
        .groupby("user_id")["purchase_time"]
        .apply(lambda x: x.diff().dt.total_seconds() / 86400)
    )
    interval_stats = intervals.groupby("user_id").agg(
        mean_interpurchase_days="mean",
        std_interpurchase_days="std",
    ).reset_index()
    agg = agg.merge(interval_stats, on="user_id", how="left")

    # Drop intermediate datetime columns
    drop_cols = ["first_purchase_time", "last_purchase_time"]
    agg.drop(columns=[c for c in drop_cols if c in agg.columns], inplace=True)

    return agg


# =====================================================================
# 3. BILLING RISK / TRANSACTION FRICTION FEATURES
# =====================================================================

def build_transaction_features(txn: pd.DataFrame, ref_date: pd.Timestamp) -> pd.DataFrame:
    """
    From: transaction_attempts table

    This is the PRIMARY signal source for involuntary churn.
    Failed payments, card issues, 3DS friction, prepaid/virtual cards, etc.

    Features:
    - total_txn_attempts, n_failed, n_success
    - failure_rate: n_failed / total → strongest invol_churn predictor
    - failure_code breakdown: n_card_declined, n_incorrect_cvc, n_expired_card, etc.
    - has_any_failure: binary
    - consecutive_failures: max streak of failures (if temporal ordering)
    - card risk signals: is_prepaid, is_virtual, cvc_fail_rate
    - 3ds_friction: required 3DS but not authenticated
    - country_mismatch: billing country != card country
    - wallet_usage: apple_pay / android_pay / none
    - n_unique_cards: tried multiple cards → billing trouble
    - card_brand (one-hot top brands)
    - card_funding type
    - days_since_last_txn, days_since_first_txn
    """
    t = txn.copy()
    t["transaction_time"] = parse_dt(t["transaction_time"])
    t["is_failed"] = t["failure_code"].notna().astype(int)
    t["is_success"] = 1 - t["is_failed"]

    # ---- Core aggregations ----
    agg = t.groupby("user_id").agg(
        total_txn_attempts=("transaction_id", "count"),
        n_failed_txn=("is_failed", "sum"),
        n_success_txn=("is_success", "sum"),
        total_txn_amount=("amount_in_usd", "sum"),
        avg_txn_amount=("amount_in_usd", "mean"),
        max_txn_amount=("amount_in_usd", "max"),
        first_txn_time=("transaction_time", "min"),
        last_txn_time=("transaction_time", "max"),
        n_unique_cards=("card_brand", "nunique"),
    ).reset_index()

    agg["txn_failure_rate"] = agg["n_failed_txn"] / agg["total_txn_attempts"]
    agg["has_any_failure"] = (agg["n_failed_txn"] > 0).astype(int)
    agg["all_failed"] = (agg["n_success_txn"] == 0).astype(int)
    agg["multiple_cards_used"] = (agg["n_unique_cards"] > 1).astype(int)

    # Recency
    agg["days_since_last_txn"] = days_between(agg["last_txn_time"], ref_date)
    agg["days_since_first_txn"] = days_between(agg["first_txn_time"], ref_date)
    agg.drop(columns=["first_txn_time", "last_txn_time"], inplace=True)

    # ---- Failure code breakdown ----
    failure_pivot = (
        t[t["is_failed"] == 1]
        .groupby(["user_id", "failure_code"])
        .size()
        .unstack(fill_value=0)
    )
    failure_pivot.columns = [f"n_fail_{c}" for c in failure_pivot.columns]
    failure_pivot = failure_pivot.reset_index()
    agg = agg.merge(failure_pivot, on="user_id", how="left")

    # ---- Card risk flags (per-user mode/max) ----
    card_risk = t.groupby("user_id").agg(
        any_prepaid=("is_prepaid", "max"),
        any_virtual=("is_virtual", "max"),
        any_business=("is_business", "max"),
        n_cvc_fail=("cvc_check", lambda x: (x == "fail").sum()),
        n_cvc_not_provided=("cvc_check", lambda x: (x == "not_provided").sum()),
        any_3ds_required=("card_3d_secure_support", lambda x: (x == "required").any()),
        n_3ds_attempted=("is_3d_secure", "sum"),
        n_3ds_authenticated=("is_3d_secure_authenticated", "sum"),
    ).reset_index()

    for col in ["any_prepaid", "any_virtual", "any_business"]:
        card_risk[col] = card_risk[col].fillna(0).astype(int)
    card_risk["any_3ds_required"] = card_risk["any_3ds_required"].fillna(False).astype(int)

    # 3DS friction: required/attempted but NOT authenticated
    card_risk["_3ds_fail_count"] = card_risk["n_3ds_attempted"] - card_risk["n_3ds_authenticated"]
    card_risk["_3ds_fail_count"] = card_risk["_3ds_fail_count"].clip(lower=0)

    agg = agg.merge(card_risk, on="user_id", how="left")

    # ---- Country mismatch ----
    mismatch = t.groupby("user_id").apply(
        lambda x: (x["billing_address_country"] != x["card_country"]).mean()
    ).reset_index(name="country_mismatch_rate")
    agg = agg.merge(mismatch, on="user_id", how="left")

    # ---- Digital wallet ----
    wallet_counts = t.groupby(["user_id", "digital_wallet"]).size().unstack(fill_value=0)
    wallet_counts.columns = [f"wallet_{c}" for c in wallet_counts.columns]
    wallet_counts = wallet_counts.reset_index()
    agg = agg.merge(wallet_counts, on="user_id", how="left")

    if "wallet_apple_pay" in agg.columns:
        agg["uses_digital_wallet"] = (
                agg.get("wallet_apple_pay", 0) + agg.get("wallet_android_pay", 0) > 0
        ).astype(int)
    else:
        agg["uses_digital_wallet"] = 0

    # ---- Card funding type (mode per user) ----
    funding_mode = t.groupby("user_id")["card_funding"].agg(
        lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else "unknown"
    ).reset_index(name="primary_card_funding")

    for f in ["debit", "credit", "prepaid"]:
        agg[f"card_funding_{f}"] = (funding_mode["primary_card_funding"] == f).astype(int)

    # ---- Card brand (mode per user) ----
    brand_mode = t.groupby("user_id")["card_brand"].agg(
        lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else "unknown"
    ).reset_index(name="primary_card_brand")

    for b in ["visa", "mc", "amex"]:
        agg[f"card_brand_{b}"] = (brand_mode["primary_card_brand"] == b).astype(int)

    # ---- Consecutive failure streaks ----
    def max_fail_streak(group):
        fails = group.sort_values("transaction_time")["is_failed"].values
        max_streak = 0
        current = 0
        for f in fails:
            if f == 1:
                current += 1
                max_streak = max(max_streak, current)
            else:
                current = 0
        return max_streak

    streaks = t.groupby("user_id").apply(max_fail_streak).reset_index(name="max_fail_streak")
    agg = agg.merge(streaks, on="user_id", how="left")

    # ---- Last txn was failure ----
    last_txn = (
        t.sort_values("transaction_time")
        .groupby("user_id")
        .tail(1)[["user_id", "is_failed"]]
        .rename(columns={"is_failed": "last_txn_failed"})
    )
    agg = agg.merge(last_txn, on="user_id", how="left")

    return agg


# =====================================================================
# 4. GENERATION / ENGAGEMENT FEATURES
# =====================================================================

def build_generation_features(gens: pd.DataFrame, ref_date: pd.Timestamp) -> pd.DataFrame:
    """
    From: generations table (user_id, generation_id, created_at, completed_at,
          failed_at, status, credit_cost, generation_type, resolution, aspect_ratio, duration)

    This captures the CORE PRODUCT USAGE — the strongest voluntary churn signal.

    Features:
    - total_generations, n_completed, n_failed, n_nsfw
    - generation_success_rate
    - total_credits_spent, avg_credits_per_gen
    - n_unique_gen_types: diversity of model usage → power user signal
    - generation_frequency: gens per day of activity
    - recency: days since last generation
    - avg_generation_duration (from duration column)
    - resolution distribution (720p, 1080p)
    - aspect_ratio distribution (9:16 for mobile/social, 16:9 for cinema)
    - time-to-complete: avg processing time (completed_at - created_at)
    - generation trend: recent vs early half comparison (engagement decay detection)
    - credit_burn_rate: credits per day
    """
    g = gens.copy()
    g["created_at"] = parse_dt(g["created_at"])
    g["completed_at"] = parse_dt(g["completed_at"])
    g["failed_at"] = parse_dt(g["failed_at"])
    g["is_completed"] = (g["status"] == "completed").astype(int)
    g["is_failed"] = g["status"].isin(["failed", "nsfw"]).astype(int)
    g["is_nsfw"] = (g["status"] == "nsfw").astype(int)

    # Processing time in seconds
    g["processing_seconds"] = (g["completed_at"] - g["created_at"]).dt.total_seconds()

    # ---- Core aggregations ----
    agg = g.groupby("user_id").agg(
        total_generations=("generation_id", "count"),
        n_completed_gens=("is_completed", "sum"),
        n_failed_gens=("is_failed", "sum"),
        n_nsfw_gens=("is_nsfw", "sum"),
        total_credits_spent=("credit_cost", "sum"),
        avg_credits_per_gen=("credit_cost", "mean"),
        max_credits_single_gen=("credit_cost", "max"),
        n_unique_gen_types=("generation_type", "nunique"),
        avg_gen_duration=("duration", "mean"),
        max_gen_duration=("duration", "max"),
        avg_processing_seconds=("processing_seconds", "mean"),
        first_gen_time=("created_at", "min"),
        last_gen_time=("created_at", "max"),
    ).reset_index()

    agg["gen_success_rate"] = agg["n_completed_gens"] / agg["total_generations"]
    agg["gen_failure_rate"] = agg["n_failed_gens"] / agg["total_generations"]
    agg["nsfw_rate"] = agg["n_nsfw_gens"] / agg["total_generations"]

    # Recency
    agg["days_since_last_gen"] = days_between(agg["last_gen_time"], ref_date)
    agg["days_since_first_gen"] = days_between(agg["first_gen_time"], ref_date)

    # Frequency
    gen_window = days_between(agg["first_gen_time"], agg["last_gen_time"]).clip(lower=0.01)
    agg["gen_frequency_per_day"] = agg["total_generations"] / gen_window

    # Credit burn rate (total credits / days active)
    active_days = days_between(agg["first_gen_time"], ref_date).clip(lower=1)
    agg["credit_burn_rate"] = agg["total_credits_spent"] / active_days

    agg.drop(columns=["first_gen_time", "last_gen_time"], inplace=True)

    # ---- Resolution distribution ----
    if "resolution" in g.columns:
        res_pivot = g.groupby(["user_id", "resolution"]).size().unstack(fill_value=0)
        res_pivot.columns = [f"n_res_{c}" for c in res_pivot.columns]
        res_pivot = res_pivot.reset_index()
        agg = agg.merge(res_pivot, on="user_id", how="left")

    # ---- Aspect ratio distribution ----
    ar_col = [c for c in g.columns if "aspect" in c.lower() or "ration" in c.lower()]
    if ar_col:
        ar_name = ar_col[0]
        ar_pivot = g.groupby(["user_id", ar_name]).size().unstack(fill_value=0)
        ar_pivot.columns = [f"n_ar_{c}" for c in ar_pivot.columns]
        ar_pivot = ar_pivot.reset_index()
        agg = agg.merge(ar_pivot, on="user_id", how="left")

    # ---- Generation type diversity (Shannon entropy) ----
    def type_entropy(group):
        counts = group["generation_type"].value_counts(normalize=True)
        return -(counts * np.log2(counts + 1e-9)).sum()

    entropy = g.groupby("user_id").apply(type_entropy).reset_index(name="gen_type_entropy")
    agg = agg.merge(entropy, on="user_id", how="left")

    # ---- Engagement trend: compare first half vs second half activity ----
    def engagement_trend(group):
        group = group.sort_values("created_at")
        n = len(group)
        if n < 2:
            return 0.0
        mid = n // 2
        first_half = mid
        second_half = n - mid
        # Ratio: >1 means increasing, <1 means declining
        return second_half / (first_half + 1e-9) - 1  # centered at 0

    trend = g.groupby("user_id").apply(engagement_trend).reset_index(name="gen_engagement_trend")
    agg = agg.merge(trend, on="user_id", how="left")

    # ---- Inter-generation intervals ----
    gen_intervals = (
        g.sort_values("created_at")
        .groupby("user_id")["created_at"]
        .apply(lambda x: x.diff().dt.total_seconds() / 3600)  # in hours
    )
    interval_agg = gen_intervals.groupby("user_id").agg(
        mean_intergen_hours="mean",
        std_intergen_hours="std",
        max_intergen_hours="max",
    ).reset_index()
    agg = agg.merge(interval_agg, on="user_id", how="left")

    # Increasing gaps = declining engagement
    def gap_trend(group):
        group = group.sort_values("created_at")
        diffs = group["created_at"].diff().dt.total_seconds()
        diffs = diffs.dropna()
        if len(diffs) < 2:
            return 0.0
        mid = len(diffs) // 2
        first = diffs.iloc[:mid].mean()
        second = diffs.iloc[mid:].mean()
        if first < 1:
            return 0.0
        return (second - first) / first  # positive = gaps increasing = bad

    gap_trends = g.groupby("user_id").apply(gap_trend).reset_index(name="gen_gap_trend")
    agg = agg.merge(gap_trends, on="user_id", how="left")

    return agg


# =====================================================================
# 5. ONBOARDING / QUIZ FEATURES
# =====================================================================

def build_quiz_features(quiz: pd.DataFrame) -> pd.DataFrame:
    """
    From: quizzes table (user_id, source, flow_type, team_size, experience,
          usage_plan, frustration, first_feature, role)

    Onboarding quiz completion and content are strong engagement predictors:
    - Users who fill out more fields → higher intent
    - source channel → acquisition quality
    - experience level → expectation calibration
    - frustration type → risk of voluntary churn
    - first_feature → product-market fit signal
    """
    q = quiz.copy()
    non_id_cols = [c for c in q.columns if c not in ["user_id", "Unnamed: 0"]]

    # Quiz completion score (fraction of fields filled)
    q["quiz_completion_score"] = q[non_id_cols].notna().sum(axis=1) / len(non_id_cols)
    q["quiz_fully_complete"] = (q["quiz_completion_score"] >= 0.9).astype(int)
    q["quiz_empty"] = (q["quiz_completion_score"] <= 0.1).astype(int)

    # ---- Source channel encoding ----
    organic_sources = {"youtube", "tiktok", "instagram", "twitter"}
    community_sources = {"ai-community", "word-of-mouth", "friends"}
    q["source_is_organic_social"] = q["source"].isin(organic_sources).astype(int)
    q["source_is_community"] = q["source"].isin(community_sources).astype(int)
    q["source_is_chatgpt"] = (q["source"] == "chatgpt").astype(int)
    q["source_is_google"] = (q["source"] == "google").astype(int)
    q["source_filled"] = q["source"].notna().astype(int)

    # ---- Experience encoding (ordinal) ----
    exp_map = {"beginner": 0, "intermediate": 1, "advanced": 2, "expert": 3}
    q["experience_level"] = q["experience"].map(exp_map).fillna(-1).astype(int)
    q["is_beginner"] = (q["experience"] == "beginner").astype(int)
    q["is_expert_or_advanced"] = q["experience"].isin(["expert", "advanced"]).astype(int)

    # ---- Team size (ordinal + buckets) ----
    solo_vals = {"solo", "1"}
    small_vals = {"small", "2-10"}
    q["is_solo_user"] = q["team_size"].isin(solo_vals).astype(int)
    q["is_team_user"] = (~q["team_size"].isin(solo_vals) & q["team_size"].notna()).astype(int)
    q["team_size_filled"] = q["team_size"].notna().astype(int)

    # ---- Usage plan encoding ----
    professional_plans = {"marketing", "freelance", "filmmaking"}
    personal_plans = {"personal", "social"}
    q["usage_is_professional"] = q["usage_plan"].isin(professional_plans).astype(int)
    q["usage_is_personal"] = q["usage_plan"].isin(personal_plans).astype(int)
    q["usage_plan_filled"] = q["usage_plan"].notna().astype(int)

    # ---- Frustration type encoding ----
    # Normalize messy values
    frust = q["frustration"].str.lower().str.strip()
    q["frust_is_cost"] = frust.str.contains("cost", na=False).astype(int)
    q["frust_is_inconsistent"] = frust.str.contains("inconsist", na=False).astype(int)
    q["frust_is_limited"] = frust.str.contains("limit", na=False).astype(int)
    q["frust_is_hard_prompt"] = frust.str.contains("prompt|hard|confus", na=False).astype(int)
    q["frustration_filled"] = q["frustration"].notna().astype(int)

    # ---- First feature interest ----
    ff = q["first_feature"].str.lower().str.strip()
    q["ff_is_video"] = ff.str.contains("video", na=False).astype(int)
    q["ff_is_commercial"] = ff.str.contains("commercial|ad", na=False).astype(int)
    q["ff_is_avatar"] = ff.str.contains("avatar|lipsync|talking", na=False).astype(int)
    q["ff_is_image"] = ff.str.contains("image|inpaint|upscale", na=False).astype(int)
    q["first_feature_filled"] = q["first_feature"].notna().astype(int)

    # ---- Role ----
    q["role_filled"] = q["role"].notna().astype(int)
    q["role_is_creator"] = (q["role"] == "creator").astype(int)
    q["role_is_professional"] = q["role"].isin(
        ["filmmaker", "designer", "marketer", "brand-owner", "founder"]
    ).astype(int)

    # Select final columns
    keep_cols = ["user_id", "quiz_completion_score", "quiz_fully_complete", "quiz_empty",
                 "source_is_organic_social", "source_is_community", "source_is_chatgpt",
                 "source_is_google", "source_filled",
                 "experience_level", "is_beginner", "is_expert_or_advanced",
                 "is_solo_user", "is_team_user", "team_size_filled",
                 "usage_is_professional", "usage_is_personal", "usage_plan_filled",
                 "frust_is_cost", "frust_is_inconsistent", "frust_is_limited",
                 "frust_is_hard_prompt", "frustration_filled",
                 "ff_is_video", "ff_is_commercial", "ff_is_avatar", "ff_is_image",
                 "first_feature_filled",
                 "role_filled", "role_is_creator", "role_is_professional"]

    return q[[c for c in keep_cols if c in q.columns]]


# =====================================================================
# 6. CROSS-TABLE INTERACTION FEATURES
# =====================================================================

def build_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derived features combining signals across tables.
    These capture nuanced patterns that single-table features miss.
    """
    out = df[["user_id"]].copy()

    # --- Spend efficiency: total spend vs total credits consumed ---
    if "total_spend" in df.columns and "total_credits_spent" in df.columns:
        out["credits_per_dollar"] = df["total_credits_spent"] / (df["total_spend"] + 1e-9)

    # --- Engagement vs payment health ---
    if "total_generations" in df.columns and "txn_failure_rate" in df.columns:
        # High engagement + payment failures → likely invol churn
        out["engaged_but_failing_pay"] = (
                (df["total_generations"] > df["total_generations"].median()) &
                (df["txn_failure_rate"] > 0.3)
        ).astype(int)

    # --- Low engagement + high spend → possible vol churn (paying but not using) ---
    if "total_generations" in df.columns and "total_spend" in df.columns:
        out["paying_but_not_using"] = (
                (df["total_generations"] < df["total_generations"].quantile(0.25)) &
                (df["total_spend"] > df["total_spend"].median())
        ).astype(int)

    # --- Onboarding quality vs engagement ---
    if "quiz_completion_score" in df.columns and "total_generations" in df.columns:
        out["good_onboarding_low_use"] = (
                (df["quiz_completion_score"] > 0.7) &
                (df["total_generations"] < df["total_generations"].quantile(0.25))
        ).astype(int)

    # --- Plan tier vs actual usage ---
    if "plan_tier" in df.columns and "total_generations" in df.columns:
        out["overpaying_for_tier"] = (
                (df["plan_tier"] >= 2) &
                (df["total_generations"] < df["total_generations"].quantile(0.25))
        ).astype(int)

    # --- New user with immediate payment failure ---
    if "sub_tenure_days" in df.columns and "has_any_failure" in df.columns:
        out["new_user_payment_fail"] = (
                (df["sub_tenure_days"] < df["sub_tenure_days"].quantile(0.25)) &
                (df["has_any_failure"] == 1)
        ).astype(int)

    # --- Cost frustration + high spend = risk ---
    if "frust_is_cost" in df.columns and "total_spend" in df.columns:
        out["cost_frustrated_high_spender"] = (
                (df["frust_is_cost"] == 1) &
                (df["total_spend"] > df["total_spend"].median())
        ).astype(int)

    # --- Expert user on basic plan = risk of outgrowing ---
    if "experience_level" in df.columns and "plan_tier" in df.columns:
        out["expert_on_basic"] = (
                (df["experience_level"] >= 2) &
                (df["plan_tier"] <= 1)
        ).astype(int)

    # Drop user_id (it's already in df)
    return out.drop(columns=["user_id"])


# =====================================================================
# MAIN PIPELINE
# =====================================================================

def build_features(
        data_dir: Path,
        mode: str = "train",
        output_path: Path | None = None,
) -> pd.DataFrame:
    """
    Main entry point. Loads all tables, builds features, merges on user_id.

    Args:
        data_dir: directory containing all CSV files
        mode: 'train' or 'test'
        output_path: optional path to save parquet

    Returns:
        DataFrame with user_id + all features (+ churn_status for train)
    """
    prefix = f"{mode}_users"
    print(f"\n{'=' * 60}")
    print(f"  Building features for: {mode.upper()}")
    print(f"  Data directory: {data_dir}")
    print(f"{'=' * 60}\n")

    # ---- Load user list (and labels if train) ----
    users_path = data_dir / f"{prefix}.csv"
    if users_path.exists():
        users = pd.read_csv(users_path)
    else:
        # Derive user list from properties or other available tables
        print(f"  [INFO] {users_path.name} not found, deriving user list from auxiliary tables")
        user_sets = []
        for suffix in ["_properties.csv", "_purchases.csv", "_transaction_attempts.csv",
                       "_generations.csv", "_quizzes.csv"]:
            p = data_dir / f"{prefix}{suffix}"
            if p.exists():
                user_sets.append(set(pd.read_csv(p)["user_id"].unique()))
        all_users = sorted(set.union(*user_sets)) if user_sets else []
        users = pd.DataFrame({"user_id": all_users})

    user_ids = users[["user_id"]].copy()
    print(f"  Users: {len(user_ids)}")

    # ---- Load auxiliary tables ----
    props = safe_load(data_dir / f"{prefix}_properties.csv")
    purchases = safe_load(data_dir / f"{prefix}_purchases.csv")
    txn = safe_load(data_dir / f"{prefix}_transaction_attempts.csv")
    gens = safe_load(data_dir / f"{prefix}_generations.csv")
    quiz = safe_load(data_dir / f"{prefix}_quizzes.csv")

    # ---- Determine reference date (max date across all tables) ----
    all_dates = []
    if props is not None:
        all_dates.extend(parse_dt(props["subscription_start_date"]).dropna().tolist())
    if purchases is not None:
        all_dates.extend(parse_dt(purchases["purchase_time"]).dropna().tolist())
    if txn is not None:
        all_dates.extend(parse_dt(txn["transaction_time"]).dropna().tolist())
    if gens is not None:
        all_dates.extend(parse_dt(gens["created_at"]).dropna().tolist())

    ref_date = max(all_dates) + pd.Timedelta(days=1) if all_dates else pd.Timestamp.now(tz="UTC")
    print(f"  Reference date: {ref_date}\n")

    # ---- Build feature groups ----
    feat = user_ids.copy()

    if props is not None:
        print("  [1/5] Building subscription features...")
        sub_feat = build_subscription_features(props, ref_date)
        feat = feat.merge(sub_feat, on="user_id", how="left")
        print(f"         → {sub_feat.shape[1] - 1} features")

    if purchases is not None:
        print("  [2/5] Building purchase features...")
        purch_feat = build_purchase_features(purchases, ref_date)
        feat = feat.merge(purch_feat, on="user_id", how="left")
        print(f"         → {purch_feat.shape[1] - 1} features")

    if txn is not None:
        print("  [3/5] Building transaction/billing features...")
        txn_feat = build_transaction_features(txn, ref_date)
        feat = feat.merge(txn_feat, on="user_id", how="left")
        print(f"         → {txn_feat.shape[1] - 1} features")

    if gens is not None:
        print("  [4/5] Building generation/engagement features...")
        gen_feat = build_generation_features(gens, ref_date)
        feat = feat.merge(gen_feat, on="user_id", how="left")
        print(f"         → {gen_feat.shape[1] - 1} features")

    if quiz is not None:
        print("  [5/5] Building quiz/onboarding features...")
        quiz_feat = build_quiz_features(quiz)
        feat = feat.merge(quiz_feat, on="user_id", how="left")
        print(f"         → {quiz_feat.shape[1] - 1} features")

    # ---- Cross-table interactions ----
    print("  [+] Building cross-table interaction features...")
    interaction_feat = build_interaction_features(feat)
    feat = pd.concat([feat, interaction_feat], axis=1)
    print(f"       → {interaction_feat.shape[1]} features")

    # ---- Attach labels if train ----
    if "churn_status" in users.columns:
        feat = feat.merge(users[["user_id", "churn_status"]], on="user_id", how="left")

    # ---- Final cleanup ----
    # Fill NaN for users that had no records in some tables
    numeric_cols = feat.select_dtypes(include=[np.number]).columns
    feat[numeric_cols] = feat[numeric_cols].fillna(0)

    print(f"\n  TOTAL features: {feat.shape[1] - 1 - ('churn_status' in feat.columns)}")
    print(f"  Final shape: {feat.shape}")

    # ---- Save ----
    if output_path:
        output_path = Path(output_path)
        if output_path.suffix == ".parquet":
            feat.to_parquet(output_path, index=False)
        else:
            feat.to_csv(output_path, index=False)
        print(f"  Saved to: {output_path}")

    return feat


# =====================================================================
# CLI
# =====================================================================

if __name__ == "__main__":
    _PROJECT_ROOT = Path(__file__).resolve().parents[2]

    parser = argparse.ArgumentParser(description="Higgsfield Churn Feature Engineering")
    parser.add_argument("--mode", choices=["train", "test"], required=True)
    parser.add_argument(
        "--data-dir", type=str, default=None,
        help="Directory with raw CSV files (default: data/raw/{mode})",
    )
    parser.add_argument(
        "--out", type=str, default=None,
        help="Output file path (.parquet or .csv, default: data/processed/features_{mode}.parquet)",
    )
    args = parser.parse_args()

    data_dir = (
        Path(args.data_dir)
        if args.data_dir
        else _PROJECT_ROOT / "data" / "raw" / args.mode
    )
    output_path = (
        Path(args.out)
        if args.out
        else _PROJECT_ROOT / "data" / "processed" / f"features_{args.mode}.parquet"
    )

    df = build_features(
        data_dir=data_dir,
        mode=args.mode,
        output_path=output_path,
    )

    # Print feature list summary
    print("\n" + "=" * 60)
    print("  FEATURE LIST")
    print("=" * 60)
    skip = {"user_id", "churn_status"}
    for i, col in enumerate(df.columns):
        if col not in skip:
            print(f"  {i:3d}. {col}")