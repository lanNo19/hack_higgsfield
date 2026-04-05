"""Run all strategy pipelines sequentially.

Pipelines with dependencies must run in order:
  P01 → standalone
  P02 → standalone (saves OOF for P03, P04)
  P03 → requires P02
  P04 → standalone (saves meta-features for P10)
  P05 → standalone
  P06 → requires P04 + P05
  P07 → standalone
  P08 → standalone
  P09 → standalone
  P10 → requires P04
  P11 → standalone (optional: requires pytorch-tabnet)
  P12 → standalone
  P13 → standalone; optionally blends with P04 if its artifacts exist (HIGH priority)

Usage:
    uv run python run_pipelines.py                          # run all
    uv run python run_pipelines.py --only 1 2 3             # run specific pipelines
    uv run python run_pipelines.py --skip 9 11              # skip slow/optional ones
    uv run python run_pipelines.py --from 4                 # start from pipeline N
    uv run python run_pipelines.py --optuna-trials 50       # Optuna trials for P02/P05/P12
"""
from __future__ import annotations

import argparse
import sys
import time
import traceback
from pathlib import Path
from typing import Callable

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent))


# Pipeline registry: number → (module_name_suffix, display_name, dependencies)
_PIPELINE_REGISTRY = {
    1:  ("lgbm_baseline",      "P01: LightGBM Baseline",             None),
    2:  ("optuna_big3",        "P02: Optuna Big Three",               None),
    3:  ("weighted_ensemble",  "P03: Weighted Average Ensemble",      [2]),
    4:  ("stacking",           "P04: Stacking Ensemble",              None),
    5:  ("hierarchical",       "P05: Hierarchical Two-Stage",         None),
    6:  ("hybrid",             "P06: Hierarchical + Flat Hybrid",     [4, 5]),
    7:  ("imbalance_ablation", "P07: Imbalance Ablation",             None),
    8:  ("feature_selection",  "P08: Feature Selection Ablation",     None),
    9:  ("seed_averaging",     "P09: Seed Averaging",                 None),
    10: ("neural_meta",        "P10: Neural Meta-Learner",            [4]),
    11: ("tabnet",             "P11: TabNet + Diversity",             None),
    12: ("ovr_specialists",    "P12: OvR Specialists",                None),
    13: ("kgmon_artifacts",   "P13: KGMON Artifact Exploitation",    None),
}


def _run_pipeline(n: int, **kwargs) -> dict | None:
    """Import and run pipeline n. Returns result dict or None on failure."""
    import importlib
    suffix, display_name, _ = _PIPELINE_REGISTRY[n]
    module_path = f"src.models.pipeline_{n:02d}_{suffix}"

    print(f"\n{'=' * 70}")
    print(f"  {display_name}")
    print(f"{'=' * 70}")

    try:
        mod = importlib.import_module(module_path)
        t0 = time.time()

        # Pass relevant kwargs to run()
        import inspect
        sig = inspect.signature(mod.run)
        run_kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}
        result = mod.run(**run_kwargs)

        elapsed = time.time() - t0
        print(f"\n  ✓ {display_name} completed in {elapsed:.1f}s")
        if isinstance(result, dict) and "macro_f1" in result:
            print(f"    macro_f1={result['macro_f1']}  logloss={result.get('logloss', '?')}")
        return result

    except FileNotFoundError as e:
        print(f"\n  ✗ {display_name} skipped — dependency missing: {e}")
        return None
    except Exception:
        print(f"\n  ✗ {display_name} FAILED:")
        traceback.print_exc()
        return None


def _check_dependencies(pipeline_nums: list[int]) -> list[int]:
    """Warn if a pipeline's dependencies aren't in the run set."""
    run_set = set(pipeline_nums)
    warnings = []
    for n in pipeline_nums:
        deps = _PIPELINE_REGISTRY[n][2]
        if deps:
            missing = [d for d in deps if d not in run_set]
            if missing:
                warnings.append((n, missing))

    if warnings:
        print("\nDependency warnings:")
        for n, missing in warnings:
            print(f"  P{n:02d} depends on P{missing} (not in run set) — may fail if artifacts missing")
    return pipeline_nums


def main():
    parser = argparse.ArgumentParser(
        description="Run Higgsfield churn prediction pipelines",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--only", nargs="+", type=int, metavar="N",
                        help="Run only these pipeline numbers")
    parser.add_argument("--skip", nargs="+", type=int, metavar="N",
                        help="Skip these pipeline numbers")
    parser.add_argument("--from", dest="from_n", type=int, metavar="N",
                        help="Start from pipeline number N")
    parser.add_argument("--optuna-trials", type=int, default=50,
                        help="Optuna trials for P02, P05, P12 (default: 50)")
    parser.add_argument("--n-seeds", type=int, default=5,
                        help="Seeds for P09 seed averaging (default: 5)")
    parser.add_argument("--list", action="store_true",
                        help="List all pipelines and exit")
    args = parser.parse_args()

    if args.list:
        print("\nAvailable pipelines:")
        for n, (_, name, deps) in _PIPELINE_REGISTRY.items():
            dep_str = f" [requires P{deps}]" if deps else ""
            print(f"  {n:2d}: {name}{dep_str}")
        return

    # Determine which pipelines to run
    all_nums = sorted(_PIPELINE_REGISTRY.keys())

    if args.only:
        to_run = sorted(args.only)
    elif args.from_n:
        to_run = [n for n in all_nums if n >= args.from_n]
    else:
        to_run = all_nums

    if args.skip:
        to_run = [n for n in to_run if n not in args.skip]

    _check_dependencies(to_run)

    print(f"\nRunning {len(to_run)} pipeline(s): {to_run}")
    print(f"Optuna trials: {args.optuna_trials} | Seeds (P09): {args.n_seeds}")

    run_kwargs = {
        "n_trials": args.optuna_trials,
        "n_seeds": args.n_seeds,
    }

    # ── Execute ────────────────────────────────────────────────────────────────
    summary = {}
    total_start = time.time()

    for n in to_run:
        result = _run_pipeline(n, **run_kwargs)
        summary[n] = result

    total_elapsed = time.time() - total_start

    # ── Summary table ──────────────────────────────────────────────────────────
    print(f"\n\n{'=' * 70}")
    print(f"  SUMMARY  (total: {total_elapsed / 60:.1f} min)")
    print(f"{'=' * 70}")
    print(f"  {'Pipeline':<40}  {'macro_f1':>10}  {'logloss':>10}  {'status':>8}")
    print(f"  {'-' * 40}  {'-' * 10}  {'-' * 10}  {'-' * 8}")

    for n, result in summary.items():
        _, display_name, _ = _PIPELINE_REGISTRY[n]
        name_short = display_name
        if result is None:
            print(f"  {name_short:<40}  {'—':>10}  {'—':>10}  {'FAILED':>8}")
        elif isinstance(result, dict):
            f1 = result.get("macro_f1", "?")
            ll = result.get("logloss", "?")
            print(f"  {name_short:<40}  {str(f1):>10}  {str(ll):>10}  {'OK':>8}")
        else:
            print(f"  {name_short:<40}  {'—':>10}  {'—':>10}  {'OK':>8}")

    print(f"\n  Results saved to: models/artifacts/results_strategy.csv")


if __name__ == "__main__":
    main()
