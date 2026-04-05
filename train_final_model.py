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
    # Try JSON file first
    if params_file.exists():
        params = json.loads(params_file.read_text())
        if "catboost" in params:
            return params["catboost"]
        log.warning("'catboost' key missing in %s — falling back to Optuna DB", params_file)

    # Fall back to Optuna SQLite DB
    db_path = ARTIFACTS / "optuna_p02.db"
    if not db_path.exists():
        raise FileNotFoundError(
            f"Neither {params_file} nor {db_path} found. Run Pipeline 2 first."
        )
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.load_study(
        study_name="p02_catboost",
        storage=f"sqlite:///{db_path}",
    )
    log.info("Loaded CatBoost study from DB: %d trials, best value=%.4f",
             len(study.trials), study.best_value)
    return study.best_params


def main(params_file: Path = DEFAULT_PARAMS_FILE) -> None:
    log.info("Loading best CatBoost params from %s", params_file)
    cat_params = load_best_params(params_file)
    log.info("Params: %s", cat_params)

    log.info("Loading full training data...")
    X, y = load_train_data()
    log.info("Dataset: %d samples, %d features", X.shape[0], X.shape[1])
    unique, counts = np.unique(y, return_counts=True)
    log.info("Class distribution: %s", dict(zip(unique.tolist(), counts.tolist())))

    # Remove params that are set explicitly to avoid duplicates
    cat_params.pop("loss_function", None)
    cat_params.pop("task_type", None)
    cat_params.pop("random_seed", None)
    cat_params.pop("eval_metric", None)

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
