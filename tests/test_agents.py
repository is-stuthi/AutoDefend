import pytest

from src.agents.base import scale_confidence, scale_confidence_descending
from src.agents.entropy_agent import EntropyAgent
from src.agents.frequency_agent import FrequencyAgent
from src.agents.statistical_agent import StatisticalAgent
from src.features.schema import WindowFeatures


def _features(**overrides) -> WindowFeatures:
    defaults = dict(
        unique_id_count=20.0,
        dominant_id_ratio=0.10,
        repetition_ratio=0.79,
        id_count_std=2.0,
        max_burst_ratio=0.02,
        id_entropy=4.20,
        payload_entropy=4.20,
        duplicate_ratio=0.35,
        payload_mean=50.0,
        payload_std=75.0,
        payload_variance=5600.0,
    )
    defaults.update(overrides)
    return WindowFeatures(**defaults)


def test_scale_confidence_clips_to_unit_interval():
    assert scale_confidence(-5, 0, 10) == 0.0
    assert scale_confidence(15, 0, 10) == 1.0
    assert scale_confidence(5, 0, 10) == pytest.approx(0.5)


def test_scale_confidence_rejects_non_increasing_range():
    with pytest.raises(ValueError):
        scale_confidence(1.0, 5.0, 5.0)


def test_scale_confidence_descending_mirrors_scale_confidence():
    assert scale_confidence_descending(10, zero_at=10, one_at=0) == 0.0
    assert scale_confidence_descending(0, zero_at=10, one_at=0) == 1.0
    assert scale_confidence_descending(5, zero_at=10, one_at=0) == pytest.approx(0.5)


def test_frequency_agent_scores_normal_traffic_low():
    result = FrequencyAgent().analyze(_features())
    assert result.confidence < 0.1


def test_frequency_agent_flags_dominant_id_flood():
    result = FrequencyAgent().analyze(_features(dominant_id_ratio=0.40))
    assert result.confidence > 0.8


def test_entropy_agent_scores_normal_traffic_low():
    result = EntropyAgent().analyze(_features())
    assert result.confidence < 0.1


def test_entropy_agent_flags_depressed_entropy_flood_signature():
    result = EntropyAgent().analyze(_features(id_entropy=3.4, payload_entropy=2.9))
    assert result.confidence > 0.7


def test_entropy_agent_flags_elevated_entropy_fuzz_signature():
    result = EntropyAgent().analyze(_features(id_entropy=5.2, payload_entropy=5.6))
    assert result.confidence > 0.8


def test_statistical_agent_scores_normal_traffic_low():
    result = StatisticalAgent().analyze(_features())
    assert result.confidence < 0.1


def test_statistical_agent_flags_high_duplicate_ratio():
    result = StatisticalAgent().analyze(_features(duplicate_ratio=0.55))
    assert result.confidence > 0.9
