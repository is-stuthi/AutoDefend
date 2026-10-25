import numpy as np
import pytest

from src.agents.calibration import calibrate_rule_thresholds
from src.agents.entropy_agent import EntropyAgent
from src.agents.frequency_agent import FrequencyAgent
from src.agents.statistical_agent import StatisticalAgent
from src.features.schema import WindowFeatures
from src.utils.config import CalibrationConfig

N_FEATURES = len(WindowFeatures.feature_names())


def _normal_matrix(seed: int = 0, n: int = 2000) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(1.0, 100.0, size=(n, N_FEATURES))


def test_calibration_is_deterministic():
    X = _normal_matrix()
    assert calibrate_rule_thresholds(X).to_dict() == calibrate_rule_thresholds(X.copy()).to_dict()


def test_calibration_matches_training_quantiles():
    X = _normal_matrix()
    names = WindowFeatures.feature_names()
    t = calibrate_rule_thresholds(X, CalibrationConfig(ramp_start_tail=0.01, ramp_end_tail=0.001))
    col = X[:, names.index("dominant_id_ratio")]
    assert t.frequency["dominant_id_low"] == pytest.approx(np.quantile(col, 0.99))
    assert t.frequency["dominant_id_high"] == pytest.approx(np.quantile(col, 0.999))
    col = X[:, names.index("id_entropy")]
    assert t.entropy["id_entropy_ceiling"] == pytest.approx(np.quantile(col, 0.01))
    assert t.entropy["id_entropy_floor"] == pytest.approx(np.quantile(col, 0.001))


def test_calibration_depends_only_on_the_training_matrix():
    """Calibration takes only training rows as input, so held-out data cannot influence it."""
    train = _normal_matrix(seed=1)
    before = calibrate_rule_thresholds(train).to_dict()
    held_out = _normal_matrix(seed=2) * 1000  # exists, but is not an argument to calibration
    assert held_out.shape[1] == train.shape[1]
    assert calibrate_rule_thresholds(train).to_dict() == before
    assert calibrate_rule_thresholds(train * 2).to_dict() != before


def test_calibrated_thresholds_build_valid_agents():
    t = calibrate_rule_thresholds(_normal_matrix())
    FrequencyAgent(**t.frequency)
    EntropyAgent(**t.entropy)
    StatisticalAgent(**t.statistical)


def test_degenerate_feature_collapses_to_a_step_without_error():
    X = _normal_matrix()
    X[:, WindowFeatures.feature_names().index("max_burst_ratio")] = 0.02
    t = calibrate_rule_thresholds(X)
    assert t.frequency["burst_high"] > t.frequency["burst_low"]


def test_rejects_empty_training_set():
    with pytest.raises(ValueError):
        calibrate_rule_thresholds(np.empty((0, N_FEATURES)))
