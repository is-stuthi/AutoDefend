"""Ablation: does fusing the rule detectors with the Random Forest add value?

Three binary (normal vs. attack) methods are scored on the **same** test
windows, with the **same** features and labels:

* ``rf_only``    - the Random Forest's 5-class prediction, collapsed to
  binary (``R`` = normal, any attack class = attack).
* ``rules_only`` - the frequency, entropy and statistical agents fused with
  the existing ``FusionEngine`` (weights renormalized to sum to 1 without
  the ML agent), flagged as attack when the fused score reaches the
  ``low`` threat threshold, i.e. ``threat_level != Normal``.
* ``fusion``     - all four agents fused with the configured weights,
  flagged by the same ``threat_level != Normal`` rule.

No weight or threshold is tuned on the test set: weights and threat
thresholds are the configured defaults, and rule thresholds come from the
training-only calibration.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from src.agents.base import DetectorResult
from src.agents.calibration import CalibratedThresholds
from src.agents.entropy_agent import EntropyAgent
from src.agents.frequency_agent import FrequencyAgent
from src.agents.ml_agent import MLAgent
from src.agents.statistical_agent import StatisticalAgent
from src.data.schema import NORMAL_LABEL
from src.evaluation.metrics import BinaryMetrics, EvaluationReport, binary_metrics, evaluate_predictions
from src.features.schema import WindowFeatures
from src.fusion.engine import FusionEngine, ThreatLevel
from src.models.random_forest import RandomForestDetector
from src.utils.config import FusionWeights, ThreatThresholds

METHODS = ("rf_only", "rules_only", "fusion")


@dataclass(frozen=True)
class AblationResult:
    """Binary predictions and metrics for every method on one shared test set."""

    n_test_windows: int
    predictions: dict[str, np.ndarray]  # method -> bool array, True = attack
    metrics: dict[str, BinaryMetrics]
    per_class_detection_rate: dict[str, dict[str, float]]  # method -> true label -> fraction flagged
    rf_multiclass_report: EvaluationReport


def rules_only_weights(weights: FusionWeights) -> FusionWeights:
    """Renormalize the fusion weights over the three rule agents (ML weight = 0)."""
    total = weights.frequency + weights.entropy + weights.statistical
    return FusionWeights(
        frequency=weights.frequency / total,
        entropy=weights.entropy / total,
        statistical=weights.statistical / total,
        machine_learning=0.0,
    )


def build_rule_agents(thresholds: CalibratedThresholds) -> list:
    """Instantiate the three rule agents with calibrated thresholds."""
    return [
        FrequencyAgent(**thresholds.frequency),
        EntropyAgent(**thresholds.entropy),
        StatisticalAgent(**thresholds.statistical),
    ]


def run_ablation(
    model: RandomForestDetector,
    thresholds: CalibratedThresholds,
    test_features: Sequence[WindowFeatures],
    test_labels: Sequence[str],
    fusion_weights: FusionWeights,
    threat_thresholds: ThreatThresholds,
) -> AblationResult:
    """Score RF-only, rules-only and fusion on exactly the same test windows."""
    if len(test_features) != len(test_labels):
        raise ValueError("test_features and test_labels must have the same length")

    rule_agents = build_rule_agents(thresholds)
    ml_agent = MLAgent(model)
    rules_engine = FusionEngine(rules_only_weights(fusion_weights), threat_thresholds)
    fusion_engine = FusionEngine(fusion_weights, threat_thresholds)

    labels = np.asarray(test_labels)
    rf_labels = np.asarray(model.predict(np.vstack([f.to_array() for f in test_features])))

    rules_flags: list[bool] = []
    fusion_flags: list[bool] = []
    ml_results = ml_agent.analyze_batch(test_features)
    for features, ml_result in zip(test_features, ml_results):
        rule_results: list[DetectorResult] = [agent.analyze(features) for agent in rule_agents]
        rules_flags.append(rules_engine.fuse(rule_results).threat_level != ThreatLevel.NORMAL)
        fusion_flags.append(fusion_engine.fuse([*rule_results, ml_result]).threat_level != ThreatLevel.NORMAL)

    predictions = {
        "rf_only": rf_labels != NORMAL_LABEL,
        "rules_only": np.asarray(rules_flags, dtype=bool),
        "fusion": np.asarray(fusion_flags, dtype=bool),
    }
    truth_is_attack = labels != NORMAL_LABEL
    metrics = {name: binary_metrics(truth_is_attack, flags) for name, flags in predictions.items()}

    per_class = {
        name: {
            str(cls): float(flags[labels == cls].mean()) for cls in sorted(np.unique(labels))
        }
        for name, flags in predictions.items()
    }

    class_labels = tuple(sorted(set(labels.tolist()) | set(rf_labels.tolist())))
    rf_report = evaluate_predictions(labels, rf_labels, labels=class_labels)
    return AblationResult(
        n_test_windows=len(labels),
        predictions=predictions,
        metrics=metrics,
        per_class_detection_rate=per_class,
        rf_multiclass_report=rf_report,
    )
