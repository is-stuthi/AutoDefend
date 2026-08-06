"""Confusion matrix visualization."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


def plot_confusion_matrix(
    matrix: np.ndarray,
    labels: tuple[str, ...],
    output_path: Path | str,
    title: str,
) -> None:
    """Render a confusion matrix heatmap and save it as a PNG.

    Args:
        matrix: Square confusion matrix (rows = true label, cols = predicted).
        labels: Class labels matching the matrix's row/column order.
        output_path: Destination PNG file path.
        title: Plot title.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    size = max(5.5, len(labels) * 1.2)
    fig, ax = plt.subplots(figsize=(size, size))
    image = ax.imshow(matrix, cmap="Blues")

    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(title, fontsize=10, wrap=True)

    threshold = matrix.max() / 2 if matrix.max() > 0 else 0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(
                j,
                i,
                str(matrix[i, j]),
                ha="center",
                va="center",
                color="white" if matrix[i, j] > threshold else "black",
            )

    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
