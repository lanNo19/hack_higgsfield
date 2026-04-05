"""Batch inference: apply trained Stage 1 + Stage 2 models to new users."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.models.train import S1_FEATURES, T_FEATURES, safe_features


def apply_zero_gen_gate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Hard-rule pre-filter applied before any model scoring.
    """
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

    df["is_zero_gen_user"] = (df.get("total_generations", 0) == 0).astype(int)
    return df


def predict_churn(
        user_df: pd.DataFrame,
        s1_model,
        s2_model,
        threshold_s1: float = 0.30,
) -> pd.DataFrame:
    """
    Two-stage batch inference returning only labels.

    Returns:
        DataFrame with columns: user_id, predicted_category
    """
    # 1. Apply hard rules
    user_df = apply_zero_gen_gate(user_df)
    needs_model = user_df["final_label"].isna()

    s1_feat = safe_features(user_df, S1_FEATURES)
    t_feat = safe_features(user_df, T_FEATURES)

    # 2. Stage 1 Inference
    if needs_model.sum() > 0:
        probs_s1 = s1_model.predict_proba(user_df.loc[needs_model, s1_feat])[:, 1]
        user_df.loc[needs_model, "churn_probability"] = probs_s1

        not_churn_mask = needs_model & (user_df["churn_probability"] < threshold_s1)
        user_df.loc[not_churn_mask, "final_label"] = "not_churned"

    # 3. Stage 2 Inference
    churn_idx = needs_model & (user_df["churn_probability"].fillna(0) >= threshold_s1)
    if churn_idx.sum() > 0:
        probs_s2 = s2_model.predict_proba(user_df.loc[churn_idx, t_feat])[:, 1]
        user_df.loc[churn_idx, "final_label"] = np.where(
            probs_s2 >= 0.5, "invol_churn", "vol_churn"
        )

    # 4. Cleanup: Keep only user_id and final_label
    # We reset the index to turn 'user_id' from an index into a regular column
    result = user_df[["final_label"]].copy()
    result.index.name = "user_id"
    result = result.reset_index()

    # Rename column to your specific requirement
    result = result.rename(columns={"final_label": "predicted_category"})

    return result