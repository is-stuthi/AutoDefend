"""Common interface and helpers shared by all detection agents."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from src.features.schema import WindowFeatures


@dataclass(frozen=True)
class DetectorResult:
    """Output of a single detector's analysis of one window.

    Attributes:
        detector_name: Identifier used by the fusion engine to look up
            this detector's weight.
        confidence: Attack confidence in [0.0, 1.0].
        details: Supporting values for explainability/debugging.
    """

    detector_name: str
    confidence: float
    details: dict[str, Any]


class DetectorAgent(ABC):
    """Base class for a single-purpose intrusion detection module."""

    name: str

    @abstractmethod
    def analyze(self, features: WindowFeatures) -> DetectorResult:
        """Score a window's feature vector and return a confidence result."""
        raise NotImplementedError


def scale_confidence(value: float, low: float, high: float) -> float:
    """Linearly map ``value`` from [low, high] to a [0.0, 1.0] confidence score.

    Values at or below ``low`` map to 0.0; values at or above ``high`` map
    to 1.0.

    Args:
        value: The raw feature value to scale.
        low: The value at/below which confidence is 0.0.
        high: The value at/above which confidence is 1.0.

    Returns:
        A confidence score clipped to [0.0, 1.0].
    """
    if high <= low:
        raise ValueError("high must exceed low")
    return float(min(max((value - low) / (high - low), 0.0), 1.0))


def scale_confidence_descending(value: float, zero_at: float, one_at: float) -> float:
    """Linearly map ``value`` to [0.0, 1.0] confidence as it falls, not rises.

    The mirror image of ``scale_confidence``: values at or above
    ``zero_at`` map to 0.0, values at or below ``one_at`` map to 1.0. Used
    for signals where an anomaly manifests as an unusually *low* value
    (e.g. depressed entropy during a flooding attack).

    Args:
        value: The raw feature value to scale.
        zero_at: The value at/above which confidence is 0.0.
        one_at: The value at/below which confidence is 1.0.

    Returns:
        A confidence score clipped to [0.0, 1.0].
    """
    if zero_at <= one_at:
        raise ValueError("zero_at must exceed one_at")
    return float(min(max((zero_at - value) / (zero_at - one_at), 0.0), 1.0))
