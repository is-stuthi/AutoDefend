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


@dataclass(frozen=True)
class BinaryMetrics:
    """Binary intrusion-detection metrics with ``attack`` as the positive class."""

    accuracy: float
    precision: float
    recall: float
    f1_score: float
    false_positive_rate: float
    false_negative_rate: float
    normal_recall: float
    tn: int
    fp: int
    fn: int
    tp: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1_score": self.f1_score,
            "false_positive_rate": self.false_positive_rate,
            "false_negative_rate": self.false_negative_rate,
            "normal_recall": self.normal_recall,
            "confusion_matrix": [[self.tn, self.fp], [self.fn, self.tp]],
            "confusion_matrix_labels": ["normal", "attack"],
        }

    @property
    def confusion_matrix(self) -> np.ndarray:
        return np.array([[self.tn, self.fp], [self.fn, self.tp]])


def binary_metrics(y_true_is_attack: Sequence[bool], y_pred_is_attack: Sequence[bool]) -> BinaryMetrics:
    """Compute attack-positive binary metrics, including FPR and FNR.

    Precision/recall/F1 are for the *attack* class (not macro-averaged).
    Rates with an empty denominator are reported as 0.0.
    """
    t = np.asarray(y_true_is_attack, dtype=bool)
    p = np.asarray(y_pred_is_attack, dtype=bool)
    if t.shape != p.shape:
        raise ValueError("y_true and y_pred must have the same length")

    tp = int(np.sum(t & p))
    tn = int(np.sum(~t & ~p))
    fp = int(np.sum(~t & p))
    fn = int(np.sum(t & ~p))

    def ratio(num: int, den: int) -> float:
        return float(num / den) if den else 0.0

    precision = ratio(tp, tp + fp)
    recall = ratio(tp, tp + fn)
    f1 = ratio(2 * precision * recall, precision + recall) if (precision + recall) else 0.0
    return BinaryMetrics(
        accuracy=ratio(tp + tn, t.size),
        precision=precision,
        recall=recall,
        f1_score=f1,
        false_positive_rate=ratio(fp, fp + tn),
        false_negative_rate=ratio(fn, fn + tp),
        normal_recall=ratio(tn, tn + fp),
        tn=tn,
        fp=fp,
        fn=fn,
        tp=tp,
    )
