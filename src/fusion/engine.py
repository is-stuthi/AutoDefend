"""Fusion engine combining detector outputs into a single threat score."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum

from src.agents.base import DetectorResult
from src.utils.config import FusionWeights, ThreatThresholds

logger = logging.getLogger(__name__)


class ThreatLevel(str, Enum):
    """Human-readable severity bucket for a fused threat score."""

    NORMAL = "Normal"
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"
    CRITICAL = "Critical"


@dataclass(frozen=True)
class FusionResult:
    """Fused output for a single window.

    Attributes:
        threat_score: Combined score in [0.0, 100.0].
        threat_level: Severity bucket derived from ``threat_score``.
        contributions: Each detector's weighted contribution to the score,
            keyed by detector name.
    """

    threat_score: float
    threat_level: ThreatLevel
    contributions: dict[str, float]


class FusionEngine:
    """Combines per-detector confidence scores into one threat score.

    Each detector's confidence (0-1) is multiplied by its configured
    weight and by 100 to produce a contribution; contributions sum to the
    final threat score. New detectors can be added without changing
    ``fuse`` as long as a weight is registered for their name in
    ``FusionWeights``.
    """

    def __init__(self, weights: FusionWeights, thresholds: ThreatThresholds) -> None:
        self._weights = weights.as_dict()
        self._thresholds = thresholds

    def fuse(self, detector_results: Sequence[DetectorResult]) -> FusionResult:
        """Combine detector results into a single threat score and level.

        Args:
            detector_results: One result per detector for a single window.

        Returns:
            The fused threat score, level, and per-detector contributions.
        """
        contributions: dict[str, float] = {}
        weighted_sum = 0.0

        for result in detector_results:
            weight = self._weights.get(result.detector_name)
            if weight is None:
                logger.warning(
                    "No fusion weight configured for detector '%s'; ignoring its output",
                    result.detector_name,
                )
                continue
            contribution = weight * result.confidence * 100.0
            contributions[result.detector_name] = contribution
            weighted_sum += contribution

        threat_score = min(max(weighted_sum, 0.0), 100.0)
        threat_level = self._classify(threat_score)
        return FusionResult(threat_score=threat_score, threat_level=threat_level, contributions=contributions)

    def _classify(self, score: float) -> ThreatLevel:
        thresholds = self._thresholds
        if score >= thresholds.critical:
            return ThreatLevel.CRITICAL
        if score >= thresholds.high:
            return ThreatLevel.HIGH
        if score >= thresholds.medium:
            return ThreatLevel.MEDIUM
        if score >= thresholds.low:
            return ThreatLevel.LOW
        return ThreatLevel.NORMAL
