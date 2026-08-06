"""Machine-learning detector backed by the trained Random Forest classifier."""

from __future__ import annotations

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
        probabilities = self._model.predict_proba(features.to_array().reshape(1, -1))[0]
        classes = list(self._model.classes_)

        normal_probability = probabilities[classes.index(NORMAL_LABEL)] if NORMAL_LABEL in classes else 0.0
        confidence = float(1.0 - normal_probability)
        predicted_label = classes[int(probabilities.argmax())]

        details: dict[str, float | str] = {"predicted_label": predicted_label}
        details.update({f"proba_{cls}": float(prob) for cls, prob in zip(classes, probabilities)})

        return DetectorResult(detector_name=self.name, confidence=confidence, details=details)
