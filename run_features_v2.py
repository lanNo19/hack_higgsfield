"""Rebuild the v2 feature matrix (force_rebuild=True)."""
from src.features.build_features_v2 import build_feature_matrix_v2

X, y = build_feature_matrix_v2("train", force_rebuild=True)
print(f"V2 feature matrix: {X.shape}")
print(f"Label distribution:\n{y.value_counts().to_string()}")
print(f"\nNew v2 columns (sample):")
v2_cols = [c for c in X.columns if c in [
    "n_consecutive_payment_failures", "last_transaction_was_failure",
    "has_ever_successfully_paid", "n_failed_last_7d", "days_from_first_fail_to_obs",
    "max_gap_days_last_30", "zero_gen_days_last_30", "days_since_last_completed_gen",
    "completion_rate_last14", "plan_renewal_proximity_days", "gens_last7_pct_of_avg",
    "trailing_silence_days",
]]
print(X[v2_cols].describe().T[["mean", "std", "min", "max"]].to_string())
