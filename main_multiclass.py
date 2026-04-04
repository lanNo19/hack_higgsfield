"""
Entry point for alternative 3-class direct pipelines (K, L, M, N).

Pipelines K / L / M — native multiclass (no cascade, single model):
  K: LightGBM   objective=multiclass
  L: XGBoost    objective=multi:softprob
  M: CatBoost   loss_function=MultiClass

Pipeline N — fixed cascade:
  - LightGBM S1 (same as Pipeline A)
  - LightGBM S2 using ALL features (not T_FEATURES only) → vol_churn has engagement signal
  - Joint (thr_s1, thr_s2) grid-search maximising 3-class macro F1

Usage:
    uv run python main_multiclass.py                    # run all K L M N
    uv run python main_multiclass.py --pipeline K       # single pipeline
    uv run python main_multiclass.py --pipeline K L M   # subset

Results are saved to models/artifacts/results_<name>.json
and the combined ranking to models/artifacts/results_multiclass.csv
"""
import argparse

from src.models.multiclass_pipelines import run_all_mc_pipelines, run_mc_pipeline
from src.utils.logger import get_logger

log = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="3-class direct churn pipelines (K–N)"
    )
    parser.add_argument(
        "--pipeline",
        nargs="+",
        choices=["K", "L", "M", "N"],
        default=None,
        help="Which pipeline(s) to run (default: all K L M N)",
    )
    args = parser.parse_args()

    if args.pipeline and len(args.pipeline) == 1:
        result = run_mc_pipeline(args.pipeline[0])
        log.info("Result: %s", result)
        print("\n" + "\n".join(f"  {k}: {v}" for k, v in result.items()))
    else:
        df = run_all_mc_pipelines(args.pipeline)
        print("\n=== Multiclass Pipeline Rankings ===")
        print(df.to_string(index=False))


if __name__ == "__main__":
    main()
