"""General statistical feature extraction over a window's payload bytes."""

from __future__ import annotations

import pandas as pd

from src.data.schema import CAN_ID_COLUMN, DATA_COLUMNS


def compute_statistical_features(window: pd.DataFrame) -> dict[str, float]:
    """Compute payload-level statistics and the duplicate-frame ratio.

    Args:
        window: A window's worth of rows from the dataset.

    Returns:
        A dict with ``duplicate_ratio``, ``payload_mean``, ``payload_std``,
        and ``payload_variance``.
    """
    payload = window[DATA_COLUMNS].to_numpy(dtype=float)
    duplicate_ratio = float(window.duplicated(subset=[CAN_ID_COLUMN, *DATA_COLUMNS]).mean())

    return {
        "duplicate_ratio": duplicate_ratio,
        "payload_mean": float(payload.mean()),
        "payload_std": float(payload.std()),
        "payload_variance": float(payload.var()),
    }
