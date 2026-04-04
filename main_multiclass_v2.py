"""
Entry point for v2 multiclass pipelines (Kv2, Lv2, Mv2, Nv2).

Requires v2 features to be built first:
    uv run python run_features_v2.py

Then:
    uv run python main_multiclass_v2.py               # all Kv2 Lv2 Mv2 Nv2
    uv run python main_multiclass_v2.py --pipeline Kv2
    uv run python main_multiclass_v2.py --pipeline Nv2  # fastest: fixed cascade

What changed vs v1 (main_multiclass.py):
  - 7 new vol_churn features (silence gaps, completed-only recency, decay rate)
  - 7 new invol_churn features (failure streaks, acute distress windows)
  - 11 noisy / collinear features removed
  - Stage 2 (Nv2) uses T_FEATURES_V2 (extended transaction features)
"""
import argparse

from src.models.multiclass_pipelines_v2 import (
    run_all_mc_pipelines_v2,
    run_mc_pipeline_v2,
)
from src.utils.logger import get_logger

log = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="V2 3-class churn pipelines (Kv2–Nv2)"
    )
    parser.add_argument(
        "--pipeline",
        nargs="+",
        choices=["Kv2", "Lv2", "Mv2", "Nv2"],
        default=None,
        help="Which pipeline(s) to run (default: all)",
    )
    args = parser.parse_args()

    if args.pipeline and len(args.pipeline) == 1:
        result = run_mc_pipeline_v2(args.pipeline[0])
        print("\n" + "\n".join(f"  {k}: {v}" for k, v in result.items()))
    else:
        df = run_all_mc_pipelines_v2(args.pipeline)
        print("\n=== V2 Multiclass Pipeline Rankings ===")
        print(df.to_string(index=False))


if __name__ == "__main__":
    main()
