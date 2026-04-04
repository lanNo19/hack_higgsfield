"""V2 training configuration: cleaned feature lists + v2 data loader.

Differences from train.py:
  - NOISE_FEATURES removed from G_FEATURES
  - New G_V2_FEATURES and T_V2_FEATURES added
  - S1_FEATURES_V2 = cleaned S1 + all new features
  - T_FEATURES_V2 = T_FEATURES + new transaction v2 features (for Stage 2)
  - load_feature_matrix_v2() reads from features_{split}_v2.parquet
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.models.train import (
    CAT_FEATURES_S1,
    CAT_FEATURES_S2,
    CAT_INT_FEATURES,
    CS_FEATURES,
    CROSS_FEATURES,
    P_FEATURES,
    PU_FEATURES,
    Q_FEATURES,
    S1_FEATURES,
    T_FEATURES,
    make_labels,
    make_splits,
    safe_features,
)
from src.utils.helpers import processed_path

# ── Features to remove (see analysis in plan) ─────────────────────────────────

NOISE_FEATURES = [
    "avg_processing_time_sec",
    "median_processing_time_sec",
    "pct_long_wait_gens",
    "dominant_usage_hour",
    "pct_gens_business_hours",
    "pct_gens_weekdays",
    "credit_cost_decimal",
    "inter_gen_regularity",
    "credit_regularity_score",
    "avg_gens_per_active_day",
    "n_credit_costing_gens",
]

# ── New features added in v2 ──────────────────────────────────────────────────

G_V2_FEATURES = [
    "max_gap_days_last_30",          # longest silence gap in last 30 days (vol signal)
    "zero_gen_days_last_30",         # calendar days with zero activity last 30d
    "days_since_last_completed_gen", # completed-only recency (failures ≠ using product)
    "completion_rate_last14",        # quality decay in last 14 days
    "plan_renewal_proximity_days",   # billing cycle position
    "gens_last7_pct_of_avg",         # normalised activity drop vs personal baseline
    "trailing_silence_days",         # consecutive days of inactivity right before obs_date
]

T_V2_FEATURES = [
    "n_consecutive_payment_failures", # longest streak without a success (invol signal)
    "last_transaction_was_failure",   # terminal payment state
    "has_ever_successfully_paid",     # cold-start vs established invol churner
    "n_failed_last_7d",               # acute 7-day payment distress
    "n_failed_last_14d",              # wider acute window
    "days_from_first_fail_to_obs",    # duration of broken payment state
    "recent_fail_concentration",      # last-7d failures / total failures
]

# ── Cleaned feature lists ──────────────────────────────────────────────────────

# G_FEATURES without noise
_G_FEATURES_CLEAN = [
    f for f in [
        "total_generations", "n_completed", "n_failed", "n_nsfw", "n_canceled",
        "n_queued_or_waiting", "completion_rate", "failure_rate_overall",
        "failure_rate_last_10", "failure_rate_delta", "nsfw_rate",
        "cancellation_rate", "pct_free_model_usage",
        "total_credits_consumed", "avg_credit_cost_per_gen", "credit_burn_rate",
        "credit_burn_acceleration", "premium_gen_ratio", "video_generation_ratio",
        "image_generation_ratio", "n_unique_generation_types",
        "video_to_image_graduation", "image_model_1_share", "avg_video_duration",
        "median_video_duration", "pct_high_resolution",
        "days_since_last_generation", "days_to_first_generation",
        "generation_span_days", "n_active_days", "active_days_fraction",
        "gens_first_7_days", "gens_last_7_days", "engagement_trajectory_ratio",
        "engagement_slope", "gens_last_14_vs_prior_14",
        "generation_frequency_daily", "generation_frequency_cv",
        "avg_inter_generation_hours", "median_inter_generation_hours",
    ]
    if f not in NOISE_FEATURES
]

S1_FEATURES_V2: list[str] = (
    P_FEATURES
    + _G_FEATURES_CLEAN
    + G_V2_FEATURES
    + PU_FEATURES
    + T_FEATURES
    + T_V2_FEATURES
    + Q_FEATURES
    + CS_FEATURES
    + CROSS_FEATURES
    + CAT_INT_FEATURES
)

# T_FEATURES_V2 is used for Stage 2 in the fixed cascade (Pipeline N-v2)
T_FEATURES_V2: list[str] = T_FEATURES + T_V2_FEATURES


# ── Data loader ───────────────────────────────────────────────────────────────

def load_feature_matrix_v2() -> tuple[pd.DataFrame, pd.Series]:
    """Load the v2 feature matrix (must have been built by run_features_v2.py)."""
    proc = processed_path()
    feat_path  = proc / "features_train_v2.parquet"
    label_path = proc / "labels_train_v2.parquet"
    if not feat_path.exists():
        raise FileNotFoundError(
            f"{feat_path} not found — run 'uv run python run_features_v2.py' first."
        )
    X = pd.read_parquet(feat_path)
    y = pd.read_parquet(label_path).squeeze()
    return X, y
