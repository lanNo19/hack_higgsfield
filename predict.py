"""Batch inference: apply trained Stage 1 + Stage 2 models to new users."""
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

from src.models.train import S1_FEATURES, T_FEATURES, safe_features


def apply_zero_gen_gate(df: pd.DataFrame) -> pd.DataFrame:
    """Hard-rule pre-filter."""
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
    Two-stage batch inference.
    Creates a dedicated output dataframe with only user_id and predicted_status,
    and saves it to prediction.csv in the same folder as this script.
    """
    # 1. Logic processing (internal temporary df)
    df = apply_zero_gen_gate(user_df)
    needs_model = df["final_label"].isna()

    s1_feat = safe_features(df, S1_FEATURES)
    t_feat = safe_features(df, T_FEATURES)

    if needs_model.sum() > 0:
        probs_s1 = s1_model.predict_proba(df.loc[needs_model, s1_feat])[:, 1]
        df.loc[needs_model, "temp_s1"] = probs_s1
        df.loc[needs_model & (df["temp_s1"] < threshold_s1), "final_label"] = "not_churned"

    churn_idx = needs_model & (df["temp_s1"].fillna(0) >= threshold_s1)
    if churn_idx.sum() > 0:
        probs_s2 = s2_model.predict_proba(df.loc[churn_idx, t_feat])[:, 1]
        df.loc[churn_idx, "final_label"] = np.where(
            probs_s2 >= 0.5, "invol_churn", "vol_churn"
        )

    # 2. CREATE ADDITIONAL DATA FRAME (The Clean Version)
    output_df = df.iloc[:, [0]].copy()
    output_df["predicted_status"] = df["final_label"]
    output_df.columns = ["user_id", "predicted_status"]

    # 3. SAVE THE RESULT TO THE SAME FOLDER AND PRINT THE PATH
    current_dir = Path(__file__).parent
    save_file = current_dir / "prediction.csv"

    output_df.to_csv(save_file, index=False)

    # This will print the exact location of the file to your terminal
    print(f"Predictions saved successfully to: {save_file.resolve()}")

    return output_df
