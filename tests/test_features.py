import numpy as np
import pandas as pd
import pytest

from src.data.schema import CAN_ID_COLUMN, DATA_COLUMNS
from src.features.entropy import compute_entropy_features, shannon_entropy
from src.features.extractor import extract_window_features
from src.features.frequency import compute_frequency_features
from src.features.schema import WindowFeatures
from src.features.statistics import compute_statistical_features


def test_shannon_entropy_of_uniform_distribution_is_maximal():
    values = np.array([1, 2, 3, 4] * 10)
    assert shannon_entropy(values) == pytest.approx(2.0)


def test_shannon_entropy_of_constant_values_is_zero():
    values = np.array([7, 7, 7, 7])
    assert shannon_entropy(values) == 0.0


def test_shannon_entropy_of_empty_array_is_zero():
    assert shannon_entropy(np.array([])) == 0.0


def test_compute_frequency_features_detects_dominant_id_flood():
    can_ids = pd.Series([1] * 90 + [2] * 5 + [3] * 5)
    features = compute_frequency_features(can_ids)

    assert features["dominant_id_ratio"] == pytest.approx(0.9)
    assert features["unique_id_count"] == 3.0
    assert features["max_burst_ratio"] == pytest.approx(0.9)


def test_compute_frequency_features_burst_ratio_ignores_non_consecutive_repeats():
    can_ids = pd.Series([1, 2] * 50)
    features = compute_frequency_features(can_ids)

    assert features["max_burst_ratio"] == pytest.approx(1.0 / 100.0)


def _make_window(rows: list[list[int]]) -> pd.DataFrame:
    columns = [CAN_ID_COLUMN, *DATA_COLUMNS]
    return pd.DataFrame(rows, columns=columns)


def test_compute_statistical_features_duplicate_ratio():
    rows = [[1, *[0] * 8]] * 5 + [[2, *[1] * 8]] * 5
    window = _make_window(rows)
    features = compute_statistical_features(window)

    assert features["duplicate_ratio"] == pytest.approx(0.8)


def test_compute_entropy_features_returns_expected_keys():
    rows = [[i % 3, *[i % 256] * 8] for i in range(10)]
    window = _make_window(rows)
    features = compute_entropy_features(window)

    assert set(features) == {"id_entropy", "payload_entropy"}
    assert features["id_entropy"] > 0.0


def test_extract_window_features_produces_full_feature_vector():
    rows = [[i % 4, *[i % 256] * 8] for i in range(20)]
    window = _make_window(rows)

    features = extract_window_features(window)

    assert isinstance(features, WindowFeatures)
    array = features.to_array()
    assert array.shape == (len(WindowFeatures.feature_names()),)
    assert np.all(np.isfinite(array))
