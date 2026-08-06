"""Rule-based detector for payload duplication and variance anomalies."""

from __future__ import annotations

from src.agents.base import DetectorAgent, DetectorResult, scale_confidence, scale_confidence_descending
from src.features.schema import WindowFeatures


class StatisticalAgent(DetectorAgent):
    """Flags windows with excessive duplicate frames or unusual payload variance.

    Complements the frequency and entropy agents. Percentile analysis of
    this dataset at window_size=100, stride=50 shows flooding-style
    attacks (DoS, Gear, RPM) resend near-identical frames, pushing
    ``duplicate_ratio`` above the normal-traffic 95th percentile (~0.43)
    and depressing ``payload_variance`` below the normal-traffic 10th
    percentile (~5100, from the payload collapsing toward a fixed
    value). A weaker high-variance signal captures the opposite case
    (RPM/Fuzzy payloads spread out above the 99th percentile, ~6500).

    Args:
        duplicate_low: Duplicate ratio at/below which the duplicate score is 0.
        duplicate_high: Duplicate ratio at/above which the duplicate score is 1.
        variance_flood_ceiling: Payload variance at/above which the
            low-variance (flood) score is 0.
        variance_flood_floor: Payload variance at/below which the
            low-variance (flood) score is 1.
        variance_high_floor: Payload variance at/below which the
            high-variance score is 0.
        variance_high_ceiling: Payload variance at/above which the
            high-variance score is 1.
    """

    name = "statistical"

    def __init__(
        self,
        duplicate_low: float = 0.43,
        duplicate_high: float = 0.55,
        variance_flood_ceiling: float = 5100.0,
        variance_flood_floor: float = 4000.0,
        variance_high_floor: float = 6500.0,
        variance_high_ceiling: float = 7200.0,
    ) -> None:
        self._duplicate_low = duplicate_low
        self._duplicate_high = duplicate_high
        self._variance_flood_ceiling = variance_flood_ceiling
        self._variance_flood_floor = variance_flood_floor
        self._variance_high_floor = variance_high_floor
        self._variance_high_ceiling = variance_high_ceiling

    def analyze(self, features: WindowFeatures) -> DetectorResult:
        duplicate_score = scale_confidence(
            features.duplicate_ratio, self._duplicate_low, self._duplicate_high
        )
        variance_flood_score = scale_confidence_descending(
            features.payload_variance, self._variance_flood_ceiling, self._variance_flood_floor
        )
        variance_high_score = scale_confidence(
            features.payload_variance, self._variance_high_floor, self._variance_high_ceiling
        )

        confidence = max(duplicate_score, variance_flood_score, variance_high_score)
        return DetectorResult(
            detector_name=self.name,
            confidence=confidence,
            details={
                "duplicate_ratio": features.duplicate_ratio,
                "payload_variance": features.payload_variance,
            },
        )
