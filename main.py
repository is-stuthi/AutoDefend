"""AutoDefend CLI entry point.

Runs the full CAN intrusion detection pipeline end-to-end: load, validate,
preprocess, generate sliding windows, extract features, train/evaluate the
Random Forest classifier, run the rule-based agents, fuse detector outputs,
and report results.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from src.evaluation.ablation import METHODS
from src.evaluation.blocked_cv import run_blocked_cv
from src.pipeline import ExperimentResult, run_pipeline
from src.utils.config import (
    FusionWeights,
    PathConfig,
    PipelineConfig,
    RandomForestConfig,
    SplitConfig,
    ThreatThresholds,
    WindowConfig,
)
from src.utils.logging_config import configure_logging

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the pipeline run."""
    parser = argparse.ArgumentParser(description="Run the AutoDefend CAN intrusion detection pipeline.")
    parser.add_argument("--data-path", type=Path, default=Path("data/Car_Hacking.csv"), help="Path to the CAN dataset CSV.")
    parser.add_argument("--window-size", type=int, default=100, help="Number of CAN frames per sliding window.")
    parser.add_argument("--stride", type=int, default=50, help="Row offset between consecutive windows.")
    parser.add_argument("--n-estimators", type=int, default=200, help="Number of trees in the Random Forest.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Fraction of each region (bursts / rows) held out for testing.")
    parser.add_argument("--split-strategy", choices=["blocked", "random_legacy"], default="blocked", help="'blocked' is leakage-safe (default). 'random_legacy' is the old LEAKY random split.")
    parser.add_argument("--blocked-cv", action="store_true", help="Run the 5-fold repeated blocked burst-level evaluation instead of the single split.")
    parser.add_argument("--include-legacy-comparison", action="store_true", help="Also run the LEAKY random split and store it in the report for comparison.")
    parser.add_argument("--model-path", type=Path, default=Path("artifacts/models/random_forest.joblib"), help="Where to save the trained model.")
    parser.add_argument("--results-dir", type=Path, default=Path("artifacts/results"), help="Where to save evaluation results.")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> PipelineConfig:
    """Translate parsed CLI arguments into a PipelineConfig."""
    return PipelineConfig(
        window=WindowConfig(size=args.window_size, stride=args.stride),
        random_forest=RandomForestConfig(n_estimators=args.n_estimators),
        split=SplitConfig(strategy=args.split_strategy, test_fraction=args.test_size, include_legacy_comparison=args.include_legacy_comparison),
        fusion_weights=FusionWeights(),
        thresholds=ThreatThresholds(),
        paths=PathConfig(data_file=args.data_path, model_file=args.model_path, results_dir=args.results_dir),
    )


def _print_experiment(title: str, exp: ExperimentResult) -> None:
    sp = exp.split_summary
    print(f"\n=== {title} ===")
    print(f"Split: {sp['strategy']} | train windows {sp['n_train_windows']} | test windows {sp['n_test_windows']}")
    print(f"Train classes: {sp['train_class_counts']}")
    print(f"Test classes:  {sp['test_class_counts']}")
    print(f"Shared frames train/test: {sp['shared_frames_between_train_and_test']} | min frame distance: {sp['min_frame_distance_train_to_test']}")
    if "bursts_shared_between_train_and_test" in sp:
        print(f"Bursts touched by train/test: {sp['bursts_touched_by_train']}/{sp['bursts_touched_by_test']} of {sp['n_bursts']} | shared: {sp['bursts_shared_between_train_and_test']}")
    rf = exp.ablation.rf_multiclass_report
    print(f"\nRF 5-class: accuracy {rf.accuracy:.4f} | macro P {rf.precision:.4f} R {rf.recall:.4f} F1 {rf.f1_score:.4f}")
    print(rf.classification_report)
    print("Binary ablation (attack = positive class):")
    print(f"{'Method':<12}{'Acc':>8}{'Prec':>8}{'Recall':>8}{'F1':>8}{'FPR':>8}{'FNR':>8}")
    for method in METHODS:
        m = exp.ablation.metrics[method]
        print(f"{method:<12}{m.accuracy:>8.4f}{m.precision:>8.4f}{m.recall:>8.4f}{m.f1_score:>8.4f}{m.false_positive_rate:>8.4f}{m.false_negative_rate:>8.4f}")


def main() -> None:
    """Run the AutoDefend pipeline and print a results summary."""
    args = parse_args()
    configure_logging(level=getattr(logging, args.log_level))

    config = build_config(args)
    logger.info(
        "Starting AutoDefend pipeline (window=%d, stride=%d, n_estimators=%d)",
        config.window.size,
        config.window.stride,
        config.random_forest.n_estimators,
    )

    if args.blocked_cv:
        run_blocked_cv(config)
        print(f"Wrote blocked_cv_results.json and blocked_cv_summary.csv to {config.paths.results_dir}")
        return

    result = run_pipeline(config)

    print("\n" + "=" * 60)
    print("AUTODEFEND PIPELINE RESULTS")
    print("=" * 60)
    _print_experiment("PRIMARY (leakage-safe)" if result.primary.split_summary["strategy"] == "blocked" else "PRIMARY (LEGACY LEAKY split)", result.primary)
    if result.legacy is not None:
        _print_experiment("LEGACY random split (LEAKY - do not cite)", result.legacy)
    print("=" * 60)


if __name__ == "__main__":
    main()
