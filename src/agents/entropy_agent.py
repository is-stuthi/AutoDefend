"""Rule-based detector for CAN ID and payload entropy anomalies."""

from __future__ import annotations

from src.agents.base import DetectorAgent, DetectorResult, scale_confidence, scale_confidence_descending
from src.features.schema import WindowFeatures


class EntropyAgent(DetectorAgent):
    """Flags windows with abnormally low or high Shannon entropy.

    Two opposite entropy signatures are targeted, based on percentile
    analysis of this dataset at window_size=100, stride=50:

    * Flooding attacks (DoS, and to a lesser extent Gear/RPM) repeat one
      CAN ID and payload far more than normal traffic, which *depresses*
      both ID entropy and payload entropy well below the normal-traffic
      1st percentile (~4.04 bits and ~3.76 bits respectively).
    * Fuzzing attacks inject near-random payloads and a wide spread of
      CAN IDs, which *elevates* both entropy measures well above the
      normal-traffic 99th percentile (~4.35 bits and ~4.59 bits).

    Args:
        id_entropy_ceiling: ID entropy at/above which the flood score is 0.
        id_entropy_floor: ID entropy at/below which the flood score is 1.
        payload_entropy_low_ceiling: Payload entropy at/above which the
            low-payload-entropy (flood) score is 0.
        payload_entropy_low_floor: Payload entropy at/below which the
            low-payload-entropy (flood) score is 1.
        payload_entropy_high_floor: Payload entropy at/below which the
            high-payload-entropy (fuzz) score is 0.
        payload_entropy_high_ceiling: Payload entropy at/above which the
            high-payload-entropy (fuzz) score is 1.
    """

    name = "entropy"

    def __init__(
        self,
        id_entropy_ceiling: float = 4.05,
        id_entropy_floor: float = 3.3,
        payload_entropy_low_ceiling: float = 3.76,
        payload_entropy_low_floor: float = 2.5,
        payload_entropy_high_floor: float = 4.6,
        payload_entropy_high_ceiling: float = 5.0,
    ) -> None:
        self._id_entropy_ceiling = id_entropy_ceiling
        self._id_entropy_floor = id_entropy_floor
        self._payload_entropy_low_ceiling = payload_entropy_low_ceiling
        self._payload_entropy_low_floor = payload_entropy_low_floor
        self._payload_entropy_high_floor = payload_entropy_high_floor
        self._payload_entropy_high_ceiling = payload_entropy_high_ceiling

    def analyze(self, features: WindowFeatures) -> DetectorResult:
        id_flood_score = scale_confidence_descending(
            features.id_entropy, self._id_entropy_ceiling, self._id_entropy_floor
        )
        payload_flood_score = scale_confidence_descending(
            features.payload_entropy, self._payload_entropy_low_ceiling, self._payload_entropy_low_floor
        )
        fuzz_score = scale_confidence(
            features.payload_entropy, self._payload_entropy_high_floor, self._payload_entropy_high_ceiling
        )

        confidence = max(id_flood_score, payload_flood_score, fuzz_score)
        return DetectorResult(
            detector_name=self.name,
            confidence=confidence,
            details={
                "id_entropy": features.id_entropy,
                "payload_entropy": features.payload_entropy,
            },
        )
