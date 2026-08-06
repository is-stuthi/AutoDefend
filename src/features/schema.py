"""Feature vector definition shared by the ML model and rule-based agents."""

from __future__ import annotations

from dataclasses import dataclass, fields

import numpy as np


@dataclass(frozen=True)
class WindowFeatures:
    """Statistical features describing a single sliding window of CAN traffic.

    Grouped by the detector that primarily consumes them, though every
    field is also fed to the Random Forest classifier as a flat vector.
    """

    # Frequency-analysis features
    unique_id_count: float
    dominant_id_ratio: float
    repetition_ratio: float
    id_count_std: float
    max_burst_ratio: float

    # Entropy-analysis features
    id_entropy: float
    payload_entropy: float

    # Statistical features
    duplicate_ratio: float
    payload_mean: float
    payload_std: float
    payload_variance: float

    def to_array(self) -> np.ndarray:
        """Flatten the feature vector into the order defined by ``feature_names``."""
        return np.array([getattr(self, name) for name in self.feature_names()], dtype=float)

    @staticmethod
    def feature_names() -> tuple[str, ...]:
        """Return feature names in a stable order matching ``to_array``."""
        return tuple(f.name for f in fields(WindowFeatures))
