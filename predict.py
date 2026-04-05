"""Batch inference: apply trained Stage 1 + Stage 2 models to new users."""
from __future__ import annotations

import numpy as np
import pandas as pd
import joblib
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
    and saves it to predictions/predictions.csv.
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

    # 3. SAVE THE RESULT TO SPECIFIED PATH
    current_dir = Path(__file__).parent
    save_dir = current_dir / "predictions"

    # Create the 'predictions' directory if it doesn't exist
    save_dir.mkdir(parents=True, exist_ok=True)

    save_file = save_dir / "predictions.csv"
    output_df.to_csv(save_file, index=False)

    print(f"Predictions saved successfully to: {save_file.resolve()}")

    return output_df


# ==========================================
# EXECUTION BLOCK
# ==========================================
if __name__ == "__main__":
    root_dir = Path(__file__).parent

    # Pointing to the specific input file you requested
    data_path = root_dir / "data" / "processed" / "labels_train.parquet"

    # YOU STILL NEED TO UPDATE THESE TO YOUR ACTUAL MODEL FILES!
    s1_model_path = root_dir / "YOUR_STAGE_1_MODEL.pkl"
    s2_model_path = root_dir / "YOUR_STAGE_2_MODEL.pkl"

    try:
        print(f"Loading user data from {data_path}...")
        user_data = pd.read_parquet(data_path)

        print("Loading models...")
        s1 = joblib.load(s1_model_path)
        s2 = joblib.load(s2_model_path)

        print("Running predictions...")
        predict_churn(user_df=user_data, s1_model=s1, s2_model=s2)

    except FileNotFoundError as e:
        print(f"\n[ERROR] File not found: {e.filename}")
        print("Please ensure your paths are correct (especially the model files).")
    except Exception as e:
        print(f"\n[ERROR] An unexpected error occurred: {e}")