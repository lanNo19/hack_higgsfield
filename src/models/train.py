"""
Training harness: data splitting and cross-validation loop shared by all pipelines.
"""
from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import average_precision_score

from src.utils.helpers import processed_path
from src.utils.logger import get_logger

log = get_logger(__name__)


def _fit_with_eval(
    model,
    X_tr: pd.DataFrame,
    y_tr: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    log_period: int = 100,
) -> None:
    """Fit with eval-set monitoring and early stopping for LGB/XGB; plain fit otherwise.

    Logs validation metric every `log_period` rounds so overfitting is visible.
    Early stopping triggers after 100 rounds without improvement.
    """
    if isinstance(model, lgb.LGBMClassifier):
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            eval_names=["val"],
            callbacks=[
                lgb.log_evaluation(period=log_period),
                lgb.early_stopping(stopping_rounds=100, verbose=True),
            ],
        )
    elif isinstance(model, xgb.XGBClassifier):
        model.set_params(early_stopping_rounds=100)
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=log_period)
    elif "eval_set" in inspect.signature(model.fit).parameters:
        # Generic handler for models that accept eval_set (e.g. TorchMLPClassifier)
        model.fit(X_tr, y_tr, eval_set=(X_val, y_val))
    else:
        model.fit(X_tr, y_tr)


# ── Feature group definitions ──────────────────────────────────────────────────

P_FEATURES = [
    "plan_ordinal", "plan_monthly_credits", "plan_monthly_cost_usd",
    "tenure_days", "subscription_start_month", "subscription_start_dayofweek",
]

G_FEATURES = [
    "total_generations", "n_completed", "n_failed", "n_nsfw", "n_canceled",
    "n_queued_or_waiting", "completion_rate", "failure_rate_overall",
    "failure_rate_last_10", "failure_rate_delta", "nsfw_rate",
    "cancellation_rate", "pct_free_model_usage", "n_credit_costing_gens",
    "total_credits_consumed", "avg_credit_cost_per_gen", "credit_burn_rate",
    "credit_burn_acceleration", "premium_gen_ratio", "video_generation_ratio",
    "image_generation_ratio", "n_unique_generation_types",
    "video_to_image_graduation", "image_model_1_share", "avg_video_duration",
    "median_video_duration", "pct_high_resolution", "avg_processing_time_sec",
    "median_processing_time_sec", "pct_long_wait_gens",
    "days_since_last_generation", "days_to_first_generation",
    "generation_span_days", "n_active_days", "active_days_fraction",
    "gens_first_7_days", "gens_last_7_days", "engagement_trajectory_ratio",
    "engagement_slope", "gens_last_14_vs_prior_14", "avg_gens_per_active_day",
    "generation_frequency_daily", "generation_frequency_cv",
    "avg_inter_generation_hours", "median_inter_generation_hours",
    "pct_gens_business_hours", "pct_gens_weekdays", "dominant_usage_hour",
]

PU_FEATURES = [
    "n_purchases_total", "n_subscription_creates", "n_subscription_updates",
    "n_credit_package_purchases", "n_upsell_purchases",
    "has_reactivation_purchase", "pct_credit_package_purchases",
    "total_purchase_dollars", "avg_purchase_dollars", "max_purchase_dollars",
    "credit_package_spend_total", "days_since_last_purchase",
    "avg_days_between_purchases", "has_plan_upgrade", "has_plan_downgrade",
    "n_plan_changes",
]

T_FEATURES = [
    "has_failed_but_no_successful_payment", "n_failed_without_matching_purchase",
    "first_transaction_was_failure", "n_distinct_amounts_attempted",
    "n_total_transaction_attempts", "n_successful_transactions",
    "n_failed_transactions", "transaction_failure_rate",
    "transaction_failure_rate_recent", "failure_rate_acceleration",
    "n_card_declined", "n_cvc_failures", "n_expired_card",
    "n_auth_required_failures", "n_processing_errors",
    "payment_retry_count", "uses_prepaid_card", "uses_virtual_card",
    "uses_business_card", "uses_digital_wallet", "pct_prepaid_transactions",
    "n_3d_secure_friction", "has_any_cvc_failure", "cvc_fail_rate",
    "n_cvc_unavailable", "total_transaction_amount_usd",
    "avg_transaction_amount_usd", "n_high_value_transactions",
    "days_since_last_failed_transaction",
    "days_since_last_successful_transaction",
    "payment_failure_timing_vs_activity", "active_during_payment_failure",
    "time_to_first_payment_issue_days", "n_failed_txns_before_success",
    "payment_resilience_score",
    "card_funding_type_int", "dominant_failure_code_int",
]

Q_FEATURES = [
    "quiz_completion_depth", "acquisition_channel_intent",
    "team_size_ordinal", "experience_ordinal", "usage_plan_commercial",
    "frustrated_cost", "frustrated_quality", "frustrated_limited",
    "frustrated_confusing", "first_feature_video", "role_commitment_score",
]

CS_FEATURES = [
    "engagement_health_score", "commitment_score",
    "rfm_recency_bin", "rfm_frequency_bin", "rfm_monetary_bin",
]

# Integer-encoded versions of string categoricals (for LGBM/XGB that need numeric input)
CAT_INT_FEATURES = [
    "country_encoded_int", "dominant_generation_type_int", "dominant_aspect_ratio_int",
    "usage_plan_encoded_int", "role_encoded_int", "first_feature_encoded_int",
    "source_encoded_int",
]

# Cross-table features computed in build_features.py but previously missing from all lists
CROSS_FEATURES = [
    "days_last_purchase_to_last_gen", "spend_per_generation",
    "plan_credit_utilization_pct", "plan_credit_surplus_deficit",
    "generation_to_purchase_ratio", "is_likely_free_tier_user",
    "generated_before_purchased", "credit_per_dollar_spent",
    "feature_expectation_mismatch", "billing_country_matches_profile",
    "dominant_card_brand_int",
]

# Stage 1 sees ALL feature groups — transaction signals predict overall churn too
S1_FEATURES = (P_FEATURES + G_FEATURES + PU_FEATURES + T_FEATURES
               + Q_FEATURES + CS_FEATURES + CROSS_FEATURES + CAT_INT_FEATURES)

# Categorical columns used by CatBoost
CAT_FEATURES_S1 = [
    "country_encoded", "dominant_generation_type", "dominant_aspect_ratio",
    "usage_plan_encoded", "role_encoded", "first_feature_encoded",
    "source_encoded", "card_funding_type", "dominant_failure_code",
]
CAT_FEATURES_S2 = ["dominant_failure_code", "card_funding_type"]


# ── Helpers ────────────────────────────────────────────────────────────────────

def safe_features(X: pd.DataFrame, wanted: list[str]) -> list[str]:
    """Return only columns that actually exist in X."""
    return [f for f in wanted if f in X.columns]


def make_labels(y: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns:
        y_binary — 0=not_churned, 1=churned (vol or invol)
        y_volInv — 0=vol_churn,   1=invol_churn  (only valid where y_binary==1)
    """
    y_binary = (y > 0).astype(int).values
    y_volInv = (y == 2).astype(int).values
    return y_binary, y_volInv


# ── Data loading + splitting ───────────────────────────────────────────────────

def load_feature_matrix() -> tuple[pd.DataFrame, pd.Series]:
    proc = processed_path()
    X = pd.read_parquet(proc / "features_train.parquet")
    y = pd.read_parquet(proc / "labels_train.parquet").squeeze()
    return X, y


def make_splits(
    X: pd.DataFrame,
    y: pd.Series,
    val_size: float = 0.10,
    test_size: float = 0.10,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame,
           pd.Series, pd.Series, pd.Series]:
    """Stratified train / val / test split. Fixed random_state ensures reproducibility."""
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    val_frac = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval,
        test_size=val_frac, stratify=y_trainval, random_state=random_state,
    )
    log.info(
        "Split sizes — train: %d  val: %d  test: %d",
        len(X_train), len(X_val), len(X_test),
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


# ── CV harness ─────────────────────────────────────────────────────────────────

def run_two_stage_cv(
    build_s1,
    build_s2,
    X: pd.DataFrame,
    y_binary: np.ndarray,
    y_volInv: np.ndarray,
    n_splits: int = 5,
    s1_features: list[str] | None = None,
    t_features: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Stratified K-fold CV producing OOF predictions for both stages.

    Args:
        build_s1:    Callable[[], classifier] for Stage 1
        build_s2:    Callable[[], classifier] for Stage 2
        X:           Full feature DataFrame (train partition only)
        y_binary:    0=not_churned, 1=churned
        y_volInv:    0=vol_churn,   1=invol_churn
        s1_features: Feature list for Stage 1 (defaults to S1_FEATURES)
        t_features:  Feature list for Stage 2 (defaults to T_FEATURES)

    Returns:
        (oof_s1, oof_s2) — OOF probability arrays, shape (len(X),)
    """
    s1_feat = safe_features(X, s1_features or S1_FEATURES)
    t_feat  = safe_features(X, t_features  or T_FEATURES)

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    oof_s1 = np.zeros(len(X))
    oof_s2 = np.zeros(len(X))

    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y_binary)):
        X_tr  = X.iloc[train_idx][s1_feat]
        X_val = X.iloc[val_idx][s1_feat]
        y_tr  = y_binary[train_idx]

        s1 = build_s1()
        _fit_with_eval(s1, X_tr, y_tr, X_val, y_binary[val_idx])
        oof_s1[val_idx] = s1.predict_proba(X_val)[:, 1]

        churn_tr  = train_idx[y_binary[train_idx] == 1]
        churn_val = val_idx[y_binary[val_idx]   == 1]
        if len(churn_tr) > 10 and len(churn_val) > 0:
            s2 = build_s2()
            _fit_with_eval(
                s2,
                X.iloc[churn_tr][t_feat], y_volInv[churn_tr],
                X.iloc[churn_val][t_feat], y_volInv[churn_val],
            )
            oof_s2[churn_val] = s2.predict_proba(
                X.iloc[churn_val][t_feat]
            )[:, 1]

        log.info(
            "Fold %d — S1 PR-AUC: %.4f",
            fold + 1,
            average_precision_score(y_binary[val_idx], oof_s1[val_idx]),
        )

    return oof_s1, oof_s2
