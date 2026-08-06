"""Sliding-window generation over CAN frames.

The dataset has no timestamps, so windows are drawn purely over row
order using a fixed size and stride (a "message-count" sliding window
rather than a time-based one).
"""

from __future__ import annotations

from collections.abc import Iterator

import pandas as pd

from src.data.schema import LABEL_COLUMN, NORMAL_LABEL


def iter_window_bounds(n_rows: int, window_size: int, stride: int) -> Iterator[tuple[int, int]]:
    """Yield (start, end) row-index bounds for each window.

    Args:
        n_rows: Total number of rows available.
        window_size: Number of consecutive messages per window.
        stride: Number of rows to advance between windows.

    Yields:
        Half-open (start, end) index pairs such that end - start == window_size.
    """
    start = 0
    while start + window_size <= n_rows:
        yield start, start + window_size
        start += stride


def iter_windows(df: pd.DataFrame, window_size: int, stride: int) -> Iterator[pd.DataFrame]:
    """Yield consecutive row-order windows of the dataset.

    Args:
        df: Preprocessed dataset, in original row order.
        window_size: Number of consecutive messages per window.
        stride: Number of rows to advance between windows.

    Yields:
        DataFrame slices of length ``window_size``.
    """
    for start, end in iter_window_bounds(len(df), window_size, stride):
        yield df.iloc[start:end]


def label_window(labels: pd.Series) -> str:
    """Assign a single label to a window from its per-row labels.

    Injected attack frames in this dataset are sparse single messages
    interspersed with normal traffic rather than contiguous runs, so a
    majority vote would label almost every window "normal" even when it
    contains malicious frames. Instead, a window is labeled with the
    most frequent non-normal label present; only if every row is normal
    is the window itself labeled normal. This matches how an IDS should
    behave: a single malicious frame is enough to flag the window.

    Args:
        labels: Per-row labels within a single window.

    Returns:
        The chosen window-level label.
    """
    attack_labels = labels[labels != NORMAL_LABEL]
    if attack_labels.empty:
        return NORMAL_LABEL
    return attack_labels.value_counts().idxmax()


def window_label(window: pd.DataFrame) -> str:
    """Convenience wrapper to derive a window's label from its DataFrame slice."""
    return label_window(window[LABEL_COLUMN])
