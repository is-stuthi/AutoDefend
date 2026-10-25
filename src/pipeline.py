"""End-to-end orchestration of the AutoDefend detection pipeline.

Order of operations (the split happens BEFORE anything is fitted):

    load -> preprocess -> split rows (burst-aware) -> window each subset
         -> extract features -> calibrate rule thresholds (train only)
         -> train Random Forest (train only)
         -> evaluate RF-only / rules-only / fusion on the same test windows

Kept separate from ``main.py`` so the pipeline can be invoked
programmatically (e.g. from tests) without going through the CLI.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.agents.calibration import CalibratedThresholds, calibrate_rule_thresholds
from src.data.loader import load_can_dataset
from src.data.preprocessing import preprocess
from src.data.schema import LABEL_COLUMN, NORMAL_LABEL
from src.data.split import BlockedSplit, blocked_split
from src.data.windowing import iter_segment_windows, iter_window_bounds, window_label
from src.evaluation.ablation import METHODS, AblationResult, run_ablation
from src.evaluation.metrics import EvaluationReport
from src.evaluation.visualization import plot_confusion_matrix
from src.features.extractor import extract_window_features
from src.features.schema import WindowFeatures
from src.models.random_forest import RandomForestDetector
from src.utils.config import PipelineConfig

logger = logging.getLogger(__name__)

METHOD_TITLES = {"rf_only": "RF-only", "rules_only": "Rules-only", "fusion": "Fusion (RF + rules)"}


@dataclass(frozen=True)
class WindowSet:
    """Windows of one subset: global start row, features and label per window."""

    starts: np.ndarray
    features: list[WindowFeatures]
    labels: np.ndarray

    def __len__(self) -> int:
        return len(self.labels)


@dataclass(frozen=True)
class ExperimentResult:
    """Everything produced by one train/calibrate/evaluate experiment."""

    split_summary: dict[str, Any]
    thresholds: CalibratedThresholds
    ablation: AblationResult
    feature_importances: dict[str, float]


@dataclass(frozen=True)
class PipelineResult:
    """Summary of a completed pipeline run."""

    primary: ExperimentResult
    legacy: ExperimentResult | None  # LEAKY random split, comparison only
    window_count: int


def run_pipeline(config: PipelineConfig) -> PipelineResult:
    """Run the leakage-safe experiment (and optionally the leaky legacy one)."""
    raw_df = load_can_dataset(config.paths.data_file)
    df = preprocess(raw_df)

    if config.split.strategy == "blocked":
        train, test, split_summary = _blocked_windows(df, config)
        primary = _run_experiment(train, test, split_summary, config, model_path=config.paths.model_file)
    else:
        primary = _run_legacy_random(df, config, model_path=config.paths.model_file)

    legacy = None
    if config.split.include_legacy_comparison and config.split.strategy == "blocked":
        legacy = _run_legacy_random(df, config)

    # Only the primary model is persisted.
    _save_results(primary, legacy, config)
    return PipelineResult(primary=primary, legacy=legacy, window_count=primary.split_summary["n_train_windows"] + primary.split_summary["n_test_windows"])


# --------------------------------------------------------------------------
# Splitting
# --------------------------------------------------------------------------
def _extract_windows(windows, cache: dict | None = None) -> WindowSet:
    """Extract features/labels per window. ``cache`` (start row -> result) only
    memoizes this pure function of the window rows; it carries no fitted state."""
    starts, feats, labels = [], [], []
    for start, window in windows:
        if cache is not None and start in cache:
            feature, label = cache[start]
        else:
            feature, label = extract_window_features(window), window_label(window)
            if cache is not None:
                cache[start] = (feature, label)
        starts.append(start)
        feats.append(feature)
        labels.append(label)
    return WindowSet(np.asarray(starts, dtype=int), feats, np.asarray(labels))


def _blocked_windows(df: pd.DataFrame, config: PipelineConfig) -> tuple[WindowSet, WindowSet, dict[str, Any]]:
    """Burst-aware split of rows, THEN windowing inside each subset."""
    size, stride = config.window.size, config.window.stride
    labels = df[LABEL_COLUMN].to_numpy()
    split = blocked_split(labels, window_size=size, test_fraction=config.split.test_fraction)

    start_time = time.monotonic()
    train = _extract_windows(iter_segment_windows(df, [(s.start, s.end) for s in split.train_segments], size, stride))
    test = _extract_windows(iter_segment_windows(df, [(s.start, s.end) for s in split.test_segments], size, stride))
    logger.info("Extracted features for %d train / %d test windows in %.1fs", len(train), len(test), time.monotonic() - start_time)

    summary = summarize_split("blocked", train, test, size, split=split, n_rows=len(df))
    return train, test, summary


def summarize_split(
    strategy: str,
    train: WindowSet,
    test: WindowSet,
    window_size: int,
    split: BlockedSplit | None = None,
    n_rows: int | None = None,
) -> dict[str, Any]:
    """Count windows and verify leakage: shared frames, shared bursts, nearest distance."""
    n_rows = n_rows if n_rows is not None else int(max(train.starts.max(), test.starts.max())) + window_size
    train_mask = _frame_mask(train.starts, window_size, n_rows)
    test_mask = _frame_mask(test.starts, window_size, n_rows)
    summary: dict[str, Any] = {
        "strategy": strategy,
        "window_size": window_size,
        "n_train_windows": len(train),
        "n_test_windows": len(test),
        "train_class_counts": _counts(train.labels),
        "test_class_counts": _counts(test.labels),
        "shared_frames_between_train_and_test": int(np.sum(train_mask & test_mask)),
        "min_frame_distance_train_to_test": _min_distance(train_mask, test.starts, window_size),
    }
    if split is not None:
        train_bursts = _bursts_touched(split, train.starts, window_size)
        test_bursts = _bursts_touched(split, test.starts, window_size)
        summary.update(
            {
                "n_bursts": len(split.bursts),
                "bursts_touched_by_train": len(train_bursts),
                "bursts_touched_by_test": len(test_bursts),
                "bursts_shared_between_train_and_test": len(train_bursts & test_bursts),
                "rows_dropped_at_boundaries": split.dropped_rows,
            }
        )
    return summary


def _counts(labels: np.ndarray) -> dict[str, int]:
    values, counts = np.unique(labels, return_counts=True)
    return {str(v): int(c) for v, c in zip(values, counts)}


def _frame_mask(starts: np.ndarray, size: int, n_rows: int) -> np.ndarray:
    mask = np.zeros(n_rows, dtype=bool)
    for s in starts:
        mask[s : s + size] = True
    return mask


def _min_distance(train_mask: np.ndarray, test_starts: np.ndarray, size: int) -> int:
    """Smallest number of frames strictly between any test window and any train frame."""
    train_idx = np.flatnonzero(train_mask)
    if train_idx.size == 0 or test_starts.size == 0:
        return -1
    pos = np.searchsorted(train_idx, test_starts)
    before = np.where(pos > 0, test_starts - train_idx[np.maximum(pos - 1, 0)] - 1, np.iinfo(np.int64).max)
    pos_after = np.searchsorted(train_idx, test_starts + size - 1)
    after = np.where(
        pos_after < train_idx.size,
        train_idx[np.minimum(pos_after, train_idx.size - 1)] - (test_starts + size - 1) - 1,
        np.iinfo(np.int64).max,
    )
    return int(min(before.min(), after.min()))


def _bursts_touched(split: BlockedSplit, starts: np.ndarray, size: int) -> set[int]:
    touched: set[int] = set()
    for burst in split.bursts:
        # A window [s, s+size) overlaps [b.start, b.end) iff s < b.end and s + size > b.start.
        if np.any((starts < burst.end) & (starts + size > burst.start)):
            touched.add(burst.index)
    return touched


# --------------------------------------------------------------------------
# Experiment (shared by the primary and the legacy split)
# --------------------------------------------------------------------------
def _run_experiment(
    train: WindowSet,
    test: WindowSet,
    split_summary: dict[str, Any],
    config: PipelineConfig,
    model_path=None,
) -> ExperimentResult:
    """Calibrate rules and fit the RF on ``train`` only, then evaluate on ``test``."""
    feature_names = WindowFeatures.feature_names()
    X_train = np.vstack([f.to_array() for f in train.features])

    # Calibration sees ONLY normal training windows.
    thresholds = calibrate_rule_thresholds(X_train[train.labels == NORMAL_LABEL], config.calibration)

    model = RandomForestDetector(
        n_estimators=config.random_forest.n_estimators,
        max_depth=config.random_forest.max_depth,
        random_state=config.random_forest.random_state,
        n_jobs=config.random_forest.n_jobs,
    )
    model.train(X_train, train.labels, feature_names)
    if model_path is not None:
        model.save_model(model_path)

    ablation = run_ablation(
        model, thresholds, test.features, test.labels, config.fusion_weights, config.thresholds
    )
    return ExperimentResult(
        split_summary=split_summary,
        thresholds=thresholds,
        ablation=ablation,
        feature_importances=model.feature_importances(),
    )


def _run_legacy_random(df: pd.DataFrame, config: PipelineConfig, model_path=None) -> ExperimentResult:
    """LEAKY / LEGACY: random stratified split over overlapping windows.

    Kept only to quantify how optimistic the original evaluation was.
    Windows overlap by ``size - stride`` frames, so train and test share frames.
    """
    size, stride = config.window.size, config.window.stride
    full = _extract_windows((s, df.iloc[s:e]) for s, e in iter_window_bounds(len(df), size, stride))
    idx = np.arange(len(full))
    train_idx, test_idx = train_test_split(
        idx, test_size=config.split.test_fraction, random_state=config.split.random_state, stratify=full.labels
    )
    train = _subset(full, np.sort(train_idx))
    test = _subset(full, np.sort(test_idx))
    summary = summarize_split("random_legacy_LEAKY", train, test, size, n_rows=len(df))
    return _run_experiment(train, test, summary, config, model_path=model_path)


def _subset(ws: WindowSet, idx: np.ndarray) -> WindowSet:
    return WindowSet(ws.starts[idx], [ws.features[i] for i in idx], ws.labels[idx])


# --------------------------------------------------------------------------
# Persistence
# --------------------------------------------------------------------------
def _experiment_payload(exp: ExperimentResult) -> dict[str, Any]:
    return {
        "split": exp.split_summary,
        "calibrated_thresholds": exp.thresholds.to_dict(),
        "rf_multiclass": exp.ablation.rf_multiclass_report.to_dict(),
        "binary_ablation": {m: exp.ablation.metrics[m].to_dict() for m in METHODS},
        "per_class_flag_rate": exp.ablation.per_class_detection_rate,
        "feature_importances": exp.feature_importances,
    }


def _save_results(primary: ExperimentResult, legacy: ExperimentResult | None, config: PipelineConfig) -> None:
    results_dir = config.paths.results_dir
    results_dir.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {
        "primary_experiment": _experiment_payload(primary),
        "config": {
            "window_size": config.window.size,
            "stride": config.window.stride,
            "test_fraction": config.split.test_fraction,
            "n_estimators": config.random_forest.n_estimators,
            "random_state": config.random_forest.random_state,
            "fusion_weights": config.fusion_weights.as_dict(),
            "feature_count": len(WindowFeatures.feature_names()),
        },
    }
    if legacy is not None:
        payload["LEGACY_LEAKY_random_split_do_not_cite"] = _experiment_payload(legacy)
    path = results_dir / "evaluation_report.json"
    path.write_text(json.dumps(payload, indent=2))
    (results_dir / "calibrated_thresholds.json").write_text(json.dumps(primary.thresholds.to_dict(), indent=2))
    logger.info("Saved evaluation results to %s", path)

    plot_confusion_matrix(
        primary.ablation.rf_multiclass_report.confusion_matrix,
        primary.ablation.rf_multiclass_report.class_labels,
        results_dir / "confusion_matrix_ml.png",
        title="Random Forest 5-class - blocked test split",
    )
    for method in METHODS:
        plot_confusion_matrix(
            primary.ablation.metrics[method].confusion_matrix,
            ("normal", "attack"),
            results_dir / f"confusion_matrix_{method}.png",
            title=f"{METHOD_TITLES[method]} - normal vs. attack (blocked test split)",
        )
