"""Dataset loading and schema validation."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src.data.schema import CAN_ID_COLUMN, DATA_COLUMNS, REQUIRED_COLUMNS

logger = logging.getLogger(__name__)


def load_can_dataset(path: Path | str) -> pd.DataFrame:
    """Load the raw CAN bus dataset from CSV and validate its schema.

    Args:
        path: Path to the Car-Hacking dataset CSV file.

    Returns:
        The raw dataset as a DataFrame, in original row order.

    Raises:
        FileNotFoundError: If the dataset file does not exist.
        ValueError: If the dataset does not match the expected schema.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path}")

    logger.info("Loading CAN dataset from %s", path)
    df = pd.read_csv(path)
    _validate_schema(df)
    logger.info("Loaded %d CAN frames", len(df))
    return df


def _validate_schema(df: pd.DataFrame) -> None:
    """Ensure the dataset has the expected columns, types, and no nulls."""
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    numeric_columns = [CAN_ID_COLUMN, *DATA_COLUMNS]
    non_numeric = [c for c in numeric_columns if not pd.api.types.is_numeric_dtype(df[c])]
    if non_numeric:
        raise ValueError(f"Expected numeric columns but found non-numeric: {non_numeric}")

    null_counts = df[REQUIRED_COLUMNS].isnull().sum()
    if null_counts.any():
        offending = null_counts[null_counts > 0].to_dict()
        raise ValueError(f"Dataset contains null values: {offending}")
