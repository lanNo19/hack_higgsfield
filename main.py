"""
Entry point for the HackNU 2026 churn prediction system.

Usage:
    uv run python main.py                  # run all pipelines
    uv run python main.py --pipeline A     # single pipeline
    uv run python main.py --pipeline E F   # subset
"""
import argparse

from src.models.pipelines import run_all_pipelines, run_pipeline
from src.utils.logger import get_logger

log = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Churn prediction pipeline runner")
    parser.add_argument(
        "--pipeline",
        nargs="+",
        choices=["A", "B", "C", "D", "E", "F"],
        default=None,
        help="Which pipeline(s) to run (default: all)",
    )
    args = parser.parse_args()

    if args.pipeline and len(args.pipeline) == 1:
        result = run_pipeline(args.pipeline[0])
        log.info("Result: %s", result)
    else:
        df = run_all_pipelines(args.pipeline)
        print("\n=== Final Rankings ===")
        print(df.to_string(index=False))


if __name__ == "__main__":
    main()
