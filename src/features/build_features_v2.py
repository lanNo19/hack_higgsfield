"""V2 feature matrix builder.

Extends the v1 pipeline (build_features.py) with:
  - Targeted invol_churn features from build_transaction_features_v2
  - Targeted vol_churn features from build_generation_features_v2
  - Removal of noisy / redundant features (processing time, time-of-day,
    synthetic-data artifacts, high-collinearity volume counts)

Saves to:  data/processed/features_{split}_v2.parquet
           data/processed/labels_{split}_v2.parquet   (same labels as v1)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.data.load_data import load_split
from src.data.preprocess import preprocess_all
from src.features.build_features import _build_composite_scores, _build_cross_table_features
from src.features.churn_features import (
    build_generation_features,
    build_properties_features,
    build_purchase_features,
    build_quiz_features,
    build_transaction_features,
)
from src.features.churn_features_v2 import (
    build_generation_features_v2,
    build_transaction_features_v2,
)
from src.utils.helpers import load_config, processed_path
from src.utils.logger import get_logger

log = get_logger(__name__)

_LABEL_MAP = {"not_churned": 0, "vol_churn": 1, "invol_churn": 2}

# Features that add noise without adding signal
NOISE_FEATURES = [
    # Platform latency — reflects server load, not user intent
    "avg_processing_time_sec",
    "median_processing_time_sec",
    "pct_long_wait_gens",
    # Time-of-day / weekday — timezone-dependent, meaningless at individual level
    "dominant_usage_hour",
    "pct_gens_business_hours",
    "pct_gens_weekdays",
    # Synthetic data artefacts — patterns exist only in generated data
    "credit_cost_decimal",
    "inter_gen_regularity",
    "credit_regularity_score",
    # High collinearity — redundant with total_generations + generation_frequency_daily
    "avg_gens_per_active_day",
    "n_credit_costing_gens",
]


def build_feature_matrix_v2(
    split: str = "train",
    force_rebuild: bool = False,
) -> tuple[pd.DataFrame, pd.Series | None]:
    """Build or load cached v2 feature matrix.

    Returns:
        (X, y) where y is None for test split.
    """
    assert split in ("train", "test")
    cfg      = load_config()
    obs_date = pd.Timestamp(cfg["observation_dates"][split], tz="UTC")

    out_dir    = processed_path()
    out_dir.mkdir(parents=True, exist_ok=True)
    feat_path  = out_dir / f"features_{split}_v2.parquet"
    label_path = out_dir / f"labels_{split}_v2.parquet"

    if feat_path.exists() and not force_rebuild:
        log.info("Loading cached v2 features from %s", feat_path)
        X = pd.read_parquet(feat_path)
        y = pd.read_parquet(label_path).squeeze() if label_path.exists() else None
        return X, y

    # ── Load & preprocess ─────────────────────────────────────────────────────
    log.info("=== Building v2 feature matrix for '%s' split ===", split)
    raw    = load_split(split)
    tables = preprocess_all(raw)

    all_users     = tables["users"]["user_id"].unique()
    all_users_idx = pd.Index(all_users, name="user_id")

    # ── V1 feature groups (unchanged) ─────────────────────────────────────────
    prop_feat = build_properties_features(tables["properties"], obs_date)
    gen_feat  = build_generation_features(tables["generations"], tables["properties"], obs_date)
    pur_feat  = build_purchase_features(tables["purchases"], obs_date)
    txn_feat  = build_transaction_features(tables["transactions"], tables["purchases"], obs_date)
    quiz_feat = build_quiz_features(tables["quizzes"])

    cross_feat = _build_cross_table_features(
        gen_feat=gen_feat,
        txn_feat=txn_feat,
        pur_feat=pur_feat,
        prop_feat=prop_feat,
        quiz_feat=quiz_feat,
        props_raw=tables["properties"],
        obs_date=obs_date,
    )

    # ── V2 feature groups (new targeted signals) ──────────────────────────────
    log.info("Building v2 generation features ...")
    gen_feat_v2 = build_generation_features_v2(
        tables["generations"], tables["properties"], obs_date
    )
    log.info("Building v2 transaction features ...")
    txn_feat_v2 = build_transaction_features_v2(
        tables["transactions"], obs_date
    )

    # ── Merge all ─────────────────────────────────────────────────────────────
    log.info("Merging all feature groups ...")
    feat_groups = [
        prop_feat, gen_feat, pur_feat, txn_feat, quiz_feat,
        cross_feat, gen_feat_v2, txn_feat_v2,
    ]
    X = pd.concat(feat_groups, axis=1).reindex(all_users_idx)

    # Drop internal helper columns
    _drop = ["_first_purchase", "_last_purchase"]
    X = X.drop(columns=[c for c in _drop if c in X.columns])

    # ── Remove noisy / redundant features ────────────────────────────────────
    cols_to_drop = [c for c in NOISE_FEATURES if c in X.columns]
    if cols_to_drop:
        log.info("Dropping %d noise features: %s", len(cols_to_drop), cols_to_drop)
    X = X.drop(columns=cols_to_drop)

    # ── Composite scores (same as v1) ─────────────────────────────────────────
    log.info("Building composite scores ...")
    scores = _build_composite_scores(X)
    X = pd.concat([X, scores], axis=1)

    # ── Integer-encode string categoricals ────────────────────────────────────
    _cat_cols = [
        "country_encoded", "dominant_generation_type", "dominant_aspect_ratio",
        "usage_plan_encoded", "role_encoded", "first_feature_encoded",
        "source_encoded", "card_funding_type", "dominant_failure_code",
        "dominant_card_brand",
    ]
    for col in _cat_cols:
        if col in X.columns and X[col].dtype == object:
            X[f"{col}_int"] = pd.factorize(X[col].fillna("unknown"))[0]

    log.info("V2 feature matrix shape: %s", X.shape)

    # ── Labels ────────────────────────────────────────────────────────────────
    y = None
    if "churn_status" in tables["users"].columns:
        label_series = (
            tables["users"].set_index("user_id")["churn_status"]
            .map(_LABEL_MAP)
            .reindex(all_users_idx)
        )
        y = label_series
        y.to_frame().to_parquet(label_path)
        log.info("Label distribution:\n%s", y.value_counts().to_string())

    X.to_parquet(feat_path)
    log.info("Saved v2 features to %s", feat_path)

    return X, y
