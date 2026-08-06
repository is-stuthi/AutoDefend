"""Shannon entropy feature extraction for CAN IDs and payload bytes."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.schema import CAN_ID_COLUMN, DATA_COLUMNS


def shannon_entropy(values: np.ndarray) -> float:
    """Compute the Shannon entropy (base 2, in bits) of a value distribution.

    Args:
        values: Array of discrete values.

    Returns:
        Entropy in bits. 0.0 for an empty array.
    """
    if values.size == 0:
        return 0.0

    _, counts = np.unique(values, return_counts=True)
    probabilities = counts / counts.sum()
    return float(-(probabilities * np.log2(probabilities)).sum())


def compute_entropy_features(window: pd.DataFrame) -> dict[str, float]:
    """Compute CAN ID entropy and payload byte entropy for a window.

    Low CAN ID entropy indicates a single ID dominating traffic (e.g. a
    flooding attack); high payload entropy indicates near-random byte
    values (e.g. a fuzzing attack).

    Args:
        window: A window's worth of rows from the dataset.

    Returns:
        A dict with ``id_entropy`` and ``payload_entropy``.
    """
    id_entropy = shannon_entropy(window[CAN_ID_COLUMN].to_numpy())
    payload_entropy = shannon_entropy(window[DATA_COLUMNS].to_numpy().ravel())
    return {"id_entropy": id_entropy, "payload_entropy": payload_entropy}
