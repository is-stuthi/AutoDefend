"""Orchestrates per-window feature extraction across all feature modules."""

from __future__ import annotations

import pandas as pd

from src.data.schema import CAN_ID_COLUMN
from src.features.entropy import compute_entropy_features
from src.features.frequency import compute_frequency_features
from src.features.schema import WindowFeatures
from src.features.statistics import compute_statistical_features


def extract_window_features(window: pd.DataFrame) -> WindowFeatures:
    """Compute the full feature vector for a single window.

    Args:
        window: A window's worth of rows from the dataset.

    Returns:
        The combined frequency, entropy, and statistical features.
    """
    frequency_features = compute_frequency_features(window[CAN_ID_COLUMN])
    entropy_features = compute_entropy_features(window)
    statistical_features = compute_statistical_features(window)

    return WindowFeatures(**frequency_features, **entropy_features, **statistical_features)
