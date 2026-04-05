"""Generate predictions on the test set using the trained CatBoost model."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

from src.utils.helpers import processed_path, root_path

CLASS_NAMES = ["not_churned", "vol_churn", "invol_churn"]
DEFAULT_MODEL = root_path() / "models" / "trained" / "catboost_final.cbm"
DEFAULT_OUT = root_path() / "predictions" / "predictions.csv"


def main(model_path: Path = DEFAULT_MODEL, out_path: Path = DEFAULT_OUT) -> None:
    # 1. Load model
    model = CatBoostClassifier().load_model(str(model_path))

    # 2. Load and prepare data
    df = pd.read_parquet(processed_path() / "features_test.parquet")
    user_ids = df.get("user_id")
    X = df.drop(columns=["user_id", "churn_status"], errors="ignore")

    # 3. Align features with training data
    if model.feature_names_ is not None:
        for f in [f for f in model.feature_names_ if f not in X.columns]:
            X[f] = 0
        X = X[model.feature_names_]

    # 4. Predict
    pred_class_idx = np.argmax(model.predict_proba(X), axis=1)

    # 5. Format and save 2-column output
    out = pd.DataFrame({
        "predicted_churn_status": [CLASS_NAMES[i] for i in pred_class_idx]
    })

    if user_ids is not None:
        out.insert(0, "user_id", user_ids.values)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"Success: {len(out)} predictions saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    main(model_path=args.model, out_path=args.out)