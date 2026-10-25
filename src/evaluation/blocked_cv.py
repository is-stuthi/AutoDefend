"""Repeated blocked burst-level evaluation within the single available recording.

Five folds; in fold k the k-th contiguous 20% of the bursts of every attack
type (and of every normal stretch) is held out. Every fold is an independent
experiment: its own split, windows, rule calibration (normal training windows
only), Random Forest, and evaluation. The only thing shared across folds is a
memo of the pure window-feature function, keyed by window start row.

This measures stability across different held-out portions of ONE recording.
It says nothing about other datasets, vehicles, attack campaigns or deployment.
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from src.data.loader import load_can_dataset
from src.data.preprocessing import preprocess
from src.data.schema import LABEL_COLUMN
from src.data.split import blocked_fold_split
from src.data.windowing import iter_segment_windows
from src.evaluation.ablation import METHODS
from src.pipeline import _extract_windows, _run_experiment, summarize_split
from src.utils.config import PipelineConfig

logger = logging.getLogger(__name__)

N_FOLDS = 5
BINARY_METRICS = ("accuracy", "precision", "recall", "f1_score", "false_positive_rate", "false_negative_rate")
MULTICLASS_METRICS = ("accuracy", "precision", "recall", "f1_score")


class LeakageError(AssertionError):
    """Raised when a fold's train and test sets overlap."""


def assert_no_leakage(summary: dict[str, Any], window_size: int) -> None:
    """Fail loudly if train/test share frames or bursts, or sit closer than one window."""
    if summary["shared_frames_between_train_and_test"] != 0:
        raise LeakageError(f"train/test share {summary['shared_frames_between_train_and_test']} frames")
    if summary["bursts_shared_between_train_and_test"] != 0:
        raise LeakageError(f"train/test share {summary['bursts_shared_between_train_and_test']} bursts")
    if summary["min_frame_distance_train_to_test"] < window_size:
        raise LeakageError(f"train/test only {summary['min_frame_distance_train_to_test']} frames apart")


def _per_class_recall(report) -> dict[str, float]:
    matrix = report.confusion_matrix
    return {
        str(label): float(matrix[i, i] / matrix[i].sum()) if matrix[i].sum() else float("nan")
        for i, label in enumerate(report.class_labels)
    }


def run_blocked_cv(config: PipelineConfig, n_folds: int = N_FOLDS) -> dict[str, Any]:
    """Run all folds and return (and save) the full result payload."""
    df = preprocess(load_can_dataset(config.paths.data_file))
    labels = df[LABEL_COLUMN].to_numpy()
    size, stride = config.window.size, config.window.stride
    cache: dict = {}

    folds: list[dict[str, Any]] = []
    for fold in range(n_folds):
        split = blocked_fold_split(labels, size, n_folds, fold)
        train = _extract_windows(iter_segment_windows(df, [(s.start, s.end) for s in split.train_segments], size, stride), cache)
        test = _extract_windows(iter_segment_windows(df, [(s.start, s.end) for s in split.test_segments], size, stride), cache)
        summary = summarize_split(f"blocked_fold_{fold + 1}_of_{n_folds}", train, test, size, split=split, n_rows=len(df))
        assert_no_leakage(summary, size)

        exp = _run_experiment(train, test, summary, config)  # fresh calibration + fresh RF per fold
        ab = exp.ablation
        folds.append(
            {
                "fold": fold + 1,
                "split": summary,
                "test_segments": [(s.start, s.end) for s in split.test_segments],
                "calibrated_thresholds": exp.thresholds.to_dict(),
                "rf_multiclass": {
                    **{m: getattr(ab.rf_multiclass_report, m) for m in MULTICLASS_METRICS},
                    "per_class_recall": _per_class_recall(ab.rf_multiclass_report),
                },
                "binary": {m: ab.metrics[m].to_dict() for m in METHODS},
                "per_class_flag_rate": ab.per_class_detection_rate,
            }
        )
        logger.info("Fold %d/%d done: RF acc %.4f", fold + 1, n_folds, folds[-1]["rf_multiclass"]["accuracy"])

    payload = {
        "description": "Blocked burst-level evaluation within the single available recording (not cross-dataset validation).",
        "configuration": {
            "n_folds": n_folds,
            "window_size": size,
            "stride": stride,
            "n_estimators": config.random_forest.n_estimators,
            "random_state": config.random_forest.random_state,
            "fusion_weights": config.fusion_weights.as_dict(),
            "calibration": {
                "ramp_start_tail": config.calibration.ramp_start_tail,
                "ramp_end_tail": config.calibration.ramp_end_tail,
            },
            "std": "sample standard deviation (ddof=1) across folds",
        },
        "folds": folds,
        "summary": _summarize(folds),
    }
    results_dir = config.paths.results_dir
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "blocked_cv_results.json").write_text(json.dumps(payload, indent=2))
    _write_csv(payload["summary"], results_dir / "blocked_cv_summary.csv", n_folds)
    return payload


def _summarize(folds: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    rows: dict[str, list[float]] = {}
    for m in MULTICLASS_METRICS:
        rows[f"rf_5class/{m}"] = [f["rf_multiclass"][m] for f in folds]
    for cls in folds[0]["rf_multiclass"]["per_class_recall"]:
        rows[f"rf_5class/recall_{cls}"] = [f["rf_multiclass"]["per_class_recall"][cls] for f in folds]
    for method in METHODS:
        for m in BINARY_METRICS:
            rows[f"{method}/{m}"] = [f["binary"][method][m] for f in folds]
    return {
        key: {
            "per_fold": vals,
            "mean": float(np.nanmean(vals)),
            "std": float(np.nanstd(vals, ddof=1)),
            "min": float(np.nanmin(vals)),
            "max": float(np.nanmax(vals)),
        }
        for key, vals in rows.items()
    }


def _write_csv(summary: dict[str, dict[str, Any]], path: Path, n_folds: int) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["method/metric", *[f"fold{i + 1}" for i in range(n_folds)], "mean", "std"])
        for key, stats in summary.items():
            writer.writerow([key, *[f"{v:.4f}" for v in stats["per_fold"]], f"{stats['mean']:.4f}", f"{stats['std']:.4f}"])
