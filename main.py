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

from src.evaluation.metrics import EvaluationReport
from src.pipeline import run_pipeline
from src.utils.config import (
    FusionWeights,
    PathConfig,
    PipelineConfig,
    RandomForestConfig,
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
    parser.add_argument("--test-size", type=float, default=0.2, help="Fraction of windows held out for evaluation.")
    parser.add_argument("--model-path", type=Path, default=Path("artifacts/models/random_forest.joblib"), help="Where to save the trained model.")
    parser.add_argument("--results-dir", type=Path, default=Path("artifacts/results"), help="Where to save evaluation results.")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> PipelineConfig:
    """Translate parsed CLI arguments into a PipelineConfig."""
    return PipelineConfig(
        window=WindowConfig(size=args.window_size, stride=args.stride),
        random_forest=RandomForestConfig(n_estimators=args.n_estimators, test_size=args.test_size),
        fusion_weights=FusionWeights(),
        thresholds=ThreatThresholds(),
        paths=PathConfig(data_file=args.data_path, model_file=args.model_path, results_dir=args.results_dir),
    )


def _print_report(title: str, report: EvaluationReport) -> None:
    print(f"\n--- {title} ---")
    print(f"Accuracy:  {report.accuracy:.4f}")
    print(f"Precision: {report.precision:.4f}")
    print(f"Recall:    {report.recall:.4f}")
    print(f"F1 Score:  {report.f1_score:.4f}")
    print(report.classification_report)


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

    result = run_pipeline(config)

    print("\n" + "=" * 60)
    print("AUTODEFEND PIPELINE RESULTS")
    print("=" * 60)
    print(f"Windows analyzed: {result.window_count}")
    _print_report("Random Forest Classifier (multi-class)", result.ml_report)
    _print_report("Fusion Engine (Normal vs. Attack)", result.fusion_report)
    print("=" * 60)


if __name__ == "__main__":
    main()
