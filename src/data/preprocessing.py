"""Preprocessing steps applied to the raw CAN dataset before windowing."""

from __future__ import annotations

import logging

import pandas as pd

from src.data.schema import CANONICAL_LABELS, LABEL_COLUMN

logger = logging.getLogger(__name__)


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and normalize the raw dataset.

    Trims incidental whitespace and normalizes inconsistent label casing
    (e.g. "gear" -> "Gear") to a canonical set. Row order is preserved
    since it is the only ordering signal available for windowing.

    Args:
        df: Raw, schema-validated dataset.

    Returns:
        A new DataFrame with normalized labels.

    Raises:
        ValueError: If a label is encountered that is not recognized.
    """
    df = df.copy()
    df[LABEL_COLUMN] = df[LABEL_COLUMN].astype(str).str.strip()

    normalized = df[LABEL_COLUMN].str.lower().map(CANONICAL_LABELS)
    unknown_mask = normalized.isna()
    if unknown_mask.any():
        unknown_values = sorted(df.loc[unknown_mask, LABEL_COLUMN].unique())
        raise ValueError(f"Encountered unrecognized label(s): {unknown_values}")

    df[LABEL_COLUMN] = normalized
    logger.info("Label distribution after preprocessing:\n%s", df[LABEL_COLUMN].value_counts().to_string())
    return df
