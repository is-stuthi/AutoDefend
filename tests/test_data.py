import pandas as pd
import pytest

from src.data.loader import _validate_schema
from src.data.preprocessing import preprocess
from src.data.schema import CAN_ID_COLUMN, DATA_COLUMNS, LABEL_COLUMN


def _valid_frame(labels: list[str]) -> pd.DataFrame:
    n = len(labels)
    data = {CAN_ID_COLUMN: range(n)}
    data.update({column: range(n) for column in DATA_COLUMNS})
    data[LABEL_COLUMN] = labels
    return pd.DataFrame(data)


def test_validate_schema_accepts_well_formed_frame():
    _validate_schema(_valid_frame(["R", "DoS"]))  # should not raise


def test_validate_schema_rejects_missing_column():
    df = _valid_frame(["R"]).drop(columns=[CAN_ID_COLUMN])
    with pytest.raises(ValueError, match="missing required columns"):
        _validate_schema(df)


def test_validate_schema_rejects_non_numeric_data_column():
    df = _valid_frame(["R", "DoS"])
    df[DATA_COLUMNS[0]] = ["a", "b"]
    with pytest.raises(ValueError, match="non-numeric"):
        _validate_schema(df)


def test_validate_schema_rejects_nulls():
    df = _valid_frame(["R", "DoS"])
    df.loc[0, CAN_ID_COLUMN] = None
    with pytest.raises(ValueError, match="null values"):
        _validate_schema(df)


def test_preprocess_normalizes_label_casing_and_whitespace():
    df = _valid_frame([" gear ", "RPM", "r"])
    result = preprocess(df)
    assert result[LABEL_COLUMN].tolist() == ["Gear", "RPM", "R"]


def test_preprocess_rejects_unknown_label():
    df = _valid_frame(["R", "not_a_real_label"])
    with pytest.raises(ValueError, match="unrecognized label"):
        preprocess(df)
