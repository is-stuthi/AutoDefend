"""Rule-based detector for CAN ID frequency and burst anomalies."""

from __future__ import annotations

from src.agents.base import DetectorAgent, DetectorResult, scale_confidence
from src.features.schema import WindowFeatures


class FrequencyAgent(DetectorAgent):
    """Flags windows dominated by a small number of repeated or bursty CAN IDs.

    Targets flooding-style attacks (DoS, Gear, RPM in the Car-Hacking
    dataset), which inject one CAN ID far more often than any single ID
    appears in normal traffic. ``repetition_ratio`` is deliberately not
    used here: percentile analysis of this dataset showed normal traffic
    is itself highly repetitive (~0.79, matching the periodic broadcast
    IDs on a real CAN bus), so it does not separate normal from attack
    windows. Default thresholds are calibrated from the normal-traffic
    percentiles at window_size=100, stride=50 (see project analysis
    notes): ``dominant_id_ratio`` rarely exceeds 0.15 and
    ``max_burst_ratio`` rarely exceeds 0.04 in normal traffic.
    """

    name = "frequency"

    def __init__(
        self,
        dominant_id_low: float = 0.15,
        dominant_id_high: float = 0.35,
        burst_low: float = 0.04,
        burst_high: float = 0.08,
    ) -> None:
        self._dominant_id_low = dominant_id_low
        self._dominant_id_high = dominant_id_high
        self._burst_low = burst_low
        self._burst_high = burst_high

    def analyze(self, features: WindowFeatures) -> DetectorResult:
        dominant_score = scale_confidence(
            features.dominant_id_ratio, self._dominant_id_low, self._dominant_id_high
        )
        burst_score = scale_confidence(features.max_burst_ratio, self._burst_low, self._burst_high)

        confidence = max(dominant_score, burst_score)
        return DetectorResult(
            detector_name=self.name,
            confidence=confidence,
            details={
                "dominant_id_ratio": features.dominant_id_ratio,
                "max_burst_ratio": features.max_burst_ratio,
                "repetition_ratio": features.repetition_ratio,
            },
        )
