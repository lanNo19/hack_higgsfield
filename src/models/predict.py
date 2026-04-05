"""Batch inference: apply trained Stage 1 + Stage 2 models to new users."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.models.train import S1_FEATURES, T_FEATURES, safe_features


def apply_zero_gen_gate(df: pd.DataFrame) -> pd.DataFrame:
    """Hard-rule pre-filter applied before any model scoring."""
    df = df.copy()
    if "final_label" not in df.columns:
        df["final_label"] = np.nan

    if "is_likely_free_tier_user" in df.columns:
        df.loc[df["is_likely_free_tier_user"] == 1, "final_label"] = "not_churned"

    if "has_failed_but_no_successful_payment" in df.columns:
        mask = (
                (df["has_failed_but_no_successful_payment"] == 1)
                & (df.get("is_likely_free_tier_user", 0) == 0)
        )
        df.loc[mask, "final_label"] = "invol_churn"

    return df


def predict_churn(
        user_df: pd.DataFrame,
        s1_model,
        s2_model,
        threshold_s1: float = 0.30,
) -> pd.DataFrame:
    """
    Returns ONLY user_id and predicted_status.
    """
    # 1. Setup working copy
    df = apply_zero_gen_gate(user_df)
    needs_model = df["final_label"].isna()

    s1_feat = safe_features(df, S1_FEATURES)
    t_feat = safe_features(df, T_FEATURES)

    # 2. Run Stage 1 (needed for logic, but probs won't be in final output)
    if needs_model.sum() > 0:
        probs_s1 = s1_model.predict_proba(df.loc[needs_model, s1_feat])[:, 1]
        df.loc[needs_model, "temp_probs"] = probs_s1

        not_churn_mask = needs_model & (df["temp_probs"] < threshold_s1)
        df.loc[not_churn_mask, "final_label"] = "not_churned"

    # 3. Run Stage 2
    churn_idx = needs_model & (df["temp_probs"].fillna(0) >= threshold_s1)
    if churn_idx.sum() > 0:
        probs_s2 = s2_model.predict_proba(df.loc[churn_idx, t_feat])[:, 1]
        df.loc[churn_idx, "final_label"] = np.where(
            probs_s2 >= 0.5, "invol_churn", "vol_churn"
        )

    # 4. THE CLEANUP: This part forces the 2-column output
    # We reset the index to get 'user_id' as a column, then filter and rename.
    df.index.name = "user_id"
    final_output = df.reset_index()[["user_id", "final_label"]]
    final_output = final_output.rename(columns={"final_label": "predicted_status"})

    return final_output