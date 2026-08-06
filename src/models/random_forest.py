"""Random Forest classifier for window-level CAN traffic classification."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier

logger = logging.getLogger(__name__)


class RandomForestDetector:
    """Wraps a scikit-learn Random Forest for multi-class CAN intrusion detection.

    Predicts one of the dataset's window labels (e.g. "R", "DoS", "Fuzzy",
    "RPM", "Gear") from a window's statistical feature vector.
    """

    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: int | None = None,
        random_state: int = 42,
        n_jobs: int = -1,
    ) -> None:
        self._model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=n_jobs,
        )
        self._feature_names: tuple[str, ...] | None = None
        self._is_fitted = False

    def train(self, X: np.ndarray, y: Sequence[str], feature_names: Sequence[str]) -> None:
        """Fit the classifier.

        Args:
            X: Feature matrix of shape (n_samples, n_features).
            y: Ground-truth labels of length n_samples.
            feature_names: Names of the columns in ``X``, used for
                explainability via ``feature_importances``.
        """
        logger.info("Training Random Forest on %d samples with %d features", X.shape[0], X.shape[1])
        self._model.fit(X, y)
        self._feature_names = tuple(feature_names)
        self._is_fitted = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict the most likely label for each row of ``X``."""
        self._ensure_fitted()
        return self._model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities for each row of ``X``."""
        self._ensure_fitted()
        return self._model.predict_proba(X)

    @property
    def classes_(self) -> np.ndarray:
        """Class labels in the order used by ``predict_proba`` columns."""
        self._ensure_fitted()
        return self._model.classes_

    def feature_importances(self) -> dict[str, float]:
        """Return Gini feature importances keyed by feature name."""
        self._ensure_fitted()
        if self._feature_names is None:
            raise RuntimeError("Feature names are unavailable")
        return dict(zip(self._feature_names, self._model.feature_importances_.tolist()))

    def save_model(self, path: Path | str) -> None:
        """Persist the trained model and its feature names to disk."""
        self._ensure_fitted()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": self._model, "feature_names": self._feature_names}, path)
        logger.info("Saved model to %s", path)

    def load_model(self, path: Path | str) -> None:
        """Load a previously trained model and its feature names from disk."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")
        payload = joblib.load(path)
        self._model = payload["model"]
        self._feature_names = payload["feature_names"]
        self._is_fitted = True
        logger.info("Loaded model from %s", path)

    def _ensure_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError("Model has not been trained or loaded yet")
