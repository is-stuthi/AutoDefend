"""Frequency-based feature extraction for CAN ID traffic within a window."""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_frequency_features(can_ids: pd.Series) -> dict[str, float]:
    """Compute CAN ID frequency, dominance, repetition, and burst features.

    Args:
        can_ids: CAN ID values for every message in a window, in order.

    Returns:
        A dict of frequency-related feature values.
    """
    n = len(can_ids)
    counts = can_ids.value_counts()

    unique_id_count = float(len(counts))
    dominant_id_ratio = float(counts.iloc[0] / n)
    repetition_ratio = float(1.0 - unique_id_count / n)
    id_count_std = float(counts.std(ddof=0)) if len(counts) > 1 else 0.0
    max_burst_ratio = float(_max_consecutive_repeats(can_ids.to_numpy()) / n)

    return {
        "unique_id_count": unique_id_count,
        "dominant_id_ratio": dominant_id_ratio,
        "repetition_ratio": repetition_ratio,
        "id_count_std": id_count_std,
        "max_burst_ratio": max_burst_ratio,
    }


def _max_consecutive_repeats(values: np.ndarray) -> int:
    """Return the length of the longest run of an identical consecutive value.

    Captures burst behaviour, e.g. a flood of identical CAN IDs sent
    back-to-back, which simple frequency counts do not distinguish from
    the same messages spread evenly across the window.
    """
    if values.size == 0:
        return 0

    changes = np.flatnonzero(values[1:] != values[:-1])
    run_boundaries = np.concatenate(([-1], changes, [values.size - 1]))
    run_lengths = np.diff(run_boundaries)
    return int(run_lengths.max())
