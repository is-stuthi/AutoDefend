"""Classification performance metrics and reporting."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)


@dataclass(frozen=True)
class EvaluationReport:
    """Aggregated classification metrics for one set of predictions."""

    accuracy: float
    precision: float
    recall: float
    f1_score: float
    confusion_matrix: np.ndarray
    class_labels: tuple[str, ...]
    classification_report: str

    def to_dict(self) -> dict[str, Any]:
        """Serialize the report to JSON-compatible primitives."""
        return {
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1_score": self.f1_score,
            "confusion_matrix": self.confusion_matrix.tolist(),
            "class_labels": list(self.class_labels),
            "classification_report": self.classification_report,
        }


def evaluate_predictions(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    labels: Sequence[str] | None = None,
) -> EvaluationReport:
    """Compute accuracy, macro precision/recall/F1, and a confusion matrix.

    Args:
        y_true: Ground-truth labels.
        y_pred: Predicted labels.
        labels: Explicit label ordering for the confusion matrix and
            report. Defaults to the sorted union of observed labels.

    Returns:
        An EvaluationReport summarizing classification performance.
    """
    resolved_labels = tuple(labels) if labels is not None else tuple(sorted(set(y_true) | set(y_pred)))

    accuracy = float(accuracy_score(y_true, y_pred))
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=resolved_labels, average="macro", zero_division=0
    )
    matrix = confusion_matrix(y_true, y_pred, labels=resolved_labels)
    report_text = classification_report(y_true, y_pred, labels=resolved_labels, zero_division=0)

    return EvaluationReport(
        accuracy=accuracy,
        precision=float(precision),
        recall=float(recall),
        f1_score=float(f1),
        confusion_matrix=matrix,
        class_labels=resolved_labels,
        classification_report=report_text,
    )
