"""
Entry point for simplified + feature-parametric multiclass pipelines.

Step 1 (auto or manual): compute SHAP feature ranking
Step 2: run model × feature-subset combinations

Models available:
  K   — LightGBM multiclass (existing, now parametric)
  L   — XGBoost multiclass  (existing, now parametric)
  M   — CatBoost multiclass (existing, now parametric)
  Ks  — LightGBM simplified (num_leaves=31, reg_lambda=3.0)
  RF  — RandomForest
  HGB — HistGradientBoosting (sklearn)
  LR  — LogisticRegression + StandardScaler

Feature subsets:
  all     — all ~139 S1_FEATURES in the parquet
  top100  — SHAP top-100
  top75   — SHAP top-75
  top50   — SHAP top-50
  top25   — SHAP top-25

Usage:
    # Full suite (all models × all feature subsets) — runs SHAP first if needed
    uv run python main_simplified.py

    # Only new simplified models, all feature subsets
    uv run python main_simplified.py --models Ks RF HGB LR

    # Existing big pipelines with reduced features only
    uv run python main_simplified.py --models K L M --features top100 top75 top50 top25

    # Single combination
    uv run python main_simplified.py --models Ks --features top50

    # Re-run SHAP ranking even if cached
    uv run python main_simplified.py --rerank
"""
import argparse

from src.models.multiclass_pipelines_simplified import ALL_MODELS, run_simplified_suite
from src.models.shap_feature_ranking import compute_shap_ranking, load_feature_lists
from src.utils.helpers import root_path
from src.utils.logger import get_logger

log = get_logger(__name__)

_FEATURE_LISTS_PATH = root_path() / "models" / "artifacts" / "feature_lists.json"
_VALID_FEATURE_KEYS = ["all", "top100", "top75", "top50", "top25"]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Simplified + feature-parametric 3-class churn pipelines"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=ALL_MODELS,
        default=None,
        help=f"Model(s) to run (default: all — {ALL_MODELS})",
    )
    parser.add_argument(
        "--features",
        nargs="+",
        choices=_VALID_FEATURE_KEYS,
        default=None,
        help="Feature subset(s) to use (default: all subsets)",
    )
    parser.add_argument(
        "--rerank",
        action="store_true",
        help="Re-run SHAP feature ranking even if feature_lists.json already exists",
    )
    args = parser.parse_args()

    # ── Step 1: ensure feature ranking exists ─────────────────────────────────
    if args.rerank or not _FEATURE_LISTS_PATH.exists():
        log.info("Computing SHAP feature ranking...")
        ranking = compute_shap_ranking()
        print("\nTop 20 features:")
        print(ranking.head(20).to_string())
    else:
        log.info("Using cached feature lists from %s", _FEATURE_LISTS_PATH)
        feature_lists = load_feature_lists()
        log.info(
            "Feature list sizes — %s",
            {k: len(v) for k, v in feature_lists.items()},
        )

    # ── Step 2: run pipelines ─────────────────────────────────────────────────
    df = run_simplified_suite(
        models=args.models,
        feature_keys=args.features,
    )

    print("\n=== Simplified Pipeline Rankings ===")
    # Show key columns only for readability
    cols = [c for c in [
        "pipeline", "weighted_f1_3class", "macro_f1_3class", "accuracy_3class",
        "f1_not_churned", "f1_vol_churn", "f1_invol_churn", "pr_auc_macro",
    ] if c in df.columns]
    print(df[cols].to_string(index=False))


if __name__ == "__main__":
    main()
