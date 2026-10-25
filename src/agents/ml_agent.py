"""Machine-learning detector backed by the trained Random Forest classifier."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from src.agents.base import DetectorAgent, DetectorResult
from src.data.schema import NORMAL_LABEL
from src.features.schema import WindowFeatures
from src.models.random_forest import RandomForestDetector


class MLAgent(DetectorAgent):
    """Wraps a trained RandomForestDetector as a fusion-compatible agent.

    Confidence is the total predicted probability mass assigned to any
    attack class (i.e. 1 - P(normal)), which keeps the score meaningful
    even though the underlying model is multi-class.
    """

    name = "machine_learning"

    def __init__(self, model: RandomForestDetector) -> None:
        self._model = model

    def analyze(self, features: WindowFeatures) -> DetectorResult:
        return self.analyze_batch([features])[0]

    def analyze_batch(self, features: Sequence[WindowFeatures]) -> list[DetectorResult]:
        """Score many windows with a single ``predict_proba`` call.

        Produces the same results as calling ``analyze`` once per window,
        without the per-call overhead of the forest.
        """
        if not features:
            return []
        probabilities = self._model.predict_proba(np.vstack([f.to_array() for f in features]))
        classes = list(self._model.classes_)
        normal_index = classes.index(NORMAL_LABEL) if NORMAL_LABEL in classes else None

        results: list[DetectorResult] = []
        for row in probabilities:
            normal_probability = row[normal_index] if normal_index is not None else 0.0
            details: dict[str, float | str] = {"predicted_label": classes[int(row.argmax())]}
            details.update({f"proba_{cls}": float(prob) for cls, prob in zip(classes, row)})
            results.append(DetectorResult(detector_name=self.name, confidence=float(1.0 - normal_probability), details=details))
        return results
