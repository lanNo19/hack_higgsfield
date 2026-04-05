"""Train final CatBoost model (P02 best weighted_f1=0.5777) on full data and save weights.

Uses best hyperparameters from models/artifacts/best_params_p02.json produced by Pipeline 2.
Trains on 100% of training data (no holdout withheld).

Usage:
    uv run python train_final_model.py
    uv run python train_final_model.py --params-file models/artifacts/best_params_p02.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from catboost import CatBoostClassifier

from src.models.pipeline_utils import ARTIFACTS, CAT_TASK_TYPE, load_train_data
from src.utils.logger import get_logger

log = get_logger(__name__)

TRAINED = Path("models/trained")
DEFAULT_PARAMS_FILE = ARTIFACTS / "best_params_p02.json"


def load_best_params(params_file: Path) -> dict:
    if not params_file.exists():
        raise FileNotFoundError(
            f"{params_file} not found. Run Pipeline 2 first (run_pipelines.py --only 2)."
        )
    params = json.loads(params_file.read_text())
    if "catboost" not in params:
        raise KeyError(f"'catboost' key not found in {params_file}. Keys: {list(params)}")
    return params["catboost"]


def main(params_file: Path = DEFAULT_PARAMS_FILE) -> None:
    log.info("Loading best CatBoost params from %s", params_file)
    cat_params = load_best_params(params_file)
    log.info("Params: %s", cat_params)

    log.info("Loading full training data...")
    X, y = load_train_data()
    log.info("Dataset: %d samples, %d features", X.shape[0], X.shape[1])
    unique, counts = np.unique(y, return_counts=True)
    log.info("Class distribution: %s", dict(zip(unique.tolist(), counts.tolist())))

    model = CatBoostClassifier(
        loss_function="MultiClass",
        task_type=CAT_TASK_TYPE,
        random_seed=42,
        verbose=100,
        **cat_params,
    )

    log.info("Training on full dataset...")
    model.fit(X, y)

    TRAINED.mkdir(parents=True, exist_ok=True)

    # Save in CatBoost native format (recommended for CatBoost)
    cb_path = TRAINED / "catboost_final.cbm"
    model.save_model(str(cb_path))
    log.info("CatBoost model saved → %s", cb_path)

    # Also save via joblib for use with sklearn-style predict_proba
    jl_path = TRAINED / "catboost_final.pkl"
    joblib.dump(model, jl_path)
    log.info("Joblib model saved  → %s", jl_path)

    # Save metadata alongside the model
    meta = {
        "pipeline": "P02_catboost",
        "weighted_f1_holdout": 0.5777,
        "macro_f1_holdout": 0.5420,
        "params": cat_params,
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "classes": ["not_churned", "vol_churn", "invol_churn"],
    }
    meta_path = TRAINED / "catboost_final_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2))
    log.info("Metadata saved      → %s", meta_path)

    log.info("Done. Final model artifacts in %s/", TRAINED)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--params-file",
        type=Path,
        default=DEFAULT_PARAMS_FILE,
        help="Path to best_params_p02.json (default: models/artifacts/best_params_p02.json)",
    )
    args = parser.parse_args()
    main(params_file=args.params_file)
