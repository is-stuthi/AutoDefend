"""End-to-end orchestration of the AutoDefend detection pipeline.

Wires together data loading, windowing, feature extraction, the trained
Random Forest, the rule-based agents, the fusion engine, and evaluation.
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

from src.agents.entropy_agent import EntropyAgent
from src.agents.frequency_agent import FrequencyAgent
from src.agents.ml_agent import MLAgent
from src.agents.statistical_agent import StatisticalAgent
from src.data.loader import load_can_dataset
from src.data.preprocessing import preprocess
from src.data.schema import NORMAL_LABEL
from src.data.windowing import iter_windows, window_label
from src.evaluation.metrics import EvaluationReport, evaluate_predictions
from src.evaluation.visualization import plot_confusion_matrix
from src.features.extractor import extract_window_features
from src.features.schema import WindowFeatures
from src.fusion.engine import FusionEngine, ThreatLevel
from src.models.random_forest import RandomForestDetector
from src.utils.config import PipelineConfig

logger = logging.getLogger(__name__)

ATTACK_LABEL = "attack"


@dataclass(frozen=True)
class PipelineResult:
    """Summary of a completed pipeline run."""

    window_count: int
    ml_report: EvaluationReport
    fusion_report: EvaluationReport
    feature_importances: dict[str, float]


def run_pipeline(config: PipelineConfig) -> PipelineResult:
    """Run the full pipeline: load, preprocess, window, extract, train, fuse, evaluate.

    Args:
        config: Fully resolved pipeline configuration.

    Returns:
        A PipelineResult summarizing classifier and fusion-engine performance.
    """
    raw_df = load_can_dataset(config.paths.data_file)
    df = preprocess(raw_df)

    features_list, labels = _build_windowed_dataset(df, config)
    feature_names = WindowFeatures.feature_names()
    X = np.vstack([f.to_array() for f in features_list])
    y = np.array(labels)

    indices = np.arange(len(features_list))
    X_train, X_test, y_train, y_test, _, idx_test = train_test_split(
        X,
        y,
        indices,
        test_size=config.random_forest.test_size,
        random_state=config.random_forest.random_state,
        stratify=y,
    )

    model = RandomForestDetector(
        n_estimators=config.random_forest.n_estimators,
        max_depth=config.random_forest.max_depth,
        random_state=config.random_forest.random_state,
        n_jobs=config.random_forest.n_jobs,
    )
    model.train(X_train, y_train, feature_names)
    model.save_model(config.paths.model_file)

    class_labels = tuple(sorted(np.unique(y).tolist()))
    y_pred = model.predict(X_test)
    ml_report = evaluate_predictions(y_test, y_pred, labels=class_labels)

    test_features = [features_list[i] for i in idx_test]
    fusion_report = _evaluate_fusion(model, test_features, y_test, config)

    _save_results(ml_report, fusion_report, model.feature_importances(), config)

    return PipelineResult(
        window_count=len(features_list),
        ml_report=ml_report,
        fusion_report=fusion_report,
        feature_importances=model.feature_importances(),
    )


def _build_windowed_dataset(df: pd.DataFrame, config: PipelineConfig) -> tuple[list[WindowFeatures], list[str]]:
    """Slide across the dataset, extracting a feature vector and label per window."""
    features_list: list[WindowFeatures] = []
    labels: list[str] = []

    start_time = time.monotonic()
    for window in iter_windows(df, config.window.size, config.window.stride):
        features_list.append(extract_window_features(window))
        labels.append(window_label(window))
    elapsed = time.monotonic() - start_time

    logger.info("Extracted features for %d windows in %.1fs", len(features_list), elapsed)
    return features_list, labels


def _evaluate_fusion(
    model: RandomForestDetector,
    test_features: list[WindowFeatures],
    test_labels: np.ndarray,
    config: PipelineConfig,
) -> EvaluationReport:
    """Run all detector agents + fusion over the held-out windows and score as binary attack detection."""
    agents = [
        FrequencyAgent(),
        EntropyAgent(),
        StatisticalAgent(),
        MLAgent(model),
    ]
    engine = FusionEngine(config.fusion_weights, config.thresholds)

    predicted_binary: list[str] = []
    for features in test_features:
        detector_results = [agent.analyze(features) for agent in agents]
        fusion_result = engine.fuse(detector_results)
        is_attack = fusion_result.threat_level != ThreatLevel.NORMAL
        predicted_binary.append(ATTACK_LABEL if is_attack else NORMAL_LABEL)

    true_binary = [NORMAL_LABEL if label == NORMAL_LABEL else ATTACK_LABEL for label in test_labels]
    return evaluate_predictions(true_binary, predicted_binary, labels=(NORMAL_LABEL, ATTACK_LABEL))


def _save_results(
    ml_report: EvaluationReport,
    fusion_report: EvaluationReport,
    feature_importances: dict[str, float],
    config: PipelineConfig,
) -> None:
    """Persist evaluation results as JSON and confusion matrix plots."""
    results_dir = config.paths.results_dir
    results_dir.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {
        "ml_classifier": ml_report.to_dict(),
        "fusion_engine": fusion_report.to_dict(),
        "feature_importances": feature_importances,
    }
    results_path = results_dir / "evaluation_report.json"
    results_path.write_text(json.dumps(payload, indent=2))
    logger.info("Saved evaluation results to %s", results_path)

    plot_confusion_matrix(
        ml_report.confusion_matrix,
        ml_report.class_labels,
        results_dir / "confusion_matrix_ml.png",
        title="Random Forest Classifier - Confusion Matrix",
    )
    plot_confusion_matrix(
        fusion_report.confusion_matrix,
        fusion_report.class_labels,
        results_dir / "confusion_matrix_fusion.png",
        title="Fusion Engine - Confusion Matrix (Normal vs. Attack)",
    )
