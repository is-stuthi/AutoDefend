import pytest

from src.agents.base import DetectorResult
from src.fusion.engine import FusionEngine, ThreatLevel
from src.utils.config import FusionWeights, ThreatThresholds


def _engine() -> FusionEngine:
    weights = FusionWeights(frequency=0.25, entropy=0.25, statistical=0.25, machine_learning=0.25)
    thresholds = ThreatThresholds(low=20.0, medium=40.0, high=60.0, critical=80.0)
    return FusionEngine(weights, thresholds)


def test_fuse_all_zero_confidence_yields_normal():
    results = [
        DetectorResult("frequency", 0.0, {}),
        DetectorResult("entropy", 0.0, {}),
        DetectorResult("statistical", 0.0, {}),
        DetectorResult("machine_learning", 0.0, {}),
    ]
    fused = _engine().fuse(results)
    assert fused.threat_score == 0.0
    assert fused.threat_level == ThreatLevel.NORMAL


def test_fuse_all_max_confidence_yields_critical():
    results = [
        DetectorResult("frequency", 1.0, {}),
        DetectorResult("entropy", 1.0, {}),
        DetectorResult("statistical", 1.0, {}),
        DetectorResult("machine_learning", 1.0, {}),
    ]
    fused = _engine().fuse(results)
    assert fused.threat_score == 100.0
    assert fused.threat_level == ThreatLevel.CRITICAL


def test_fuse_respects_threshold_boundaries():
    # Each weight is 0.25, so a single detector at full confidence contributes 25 points.
    results = [DetectorResult("machine_learning", 1.0, {})]
    fused = _engine().fuse(results)
    assert fused.threat_score == 25.0
    assert fused.threat_level == ThreatLevel.LOW


def test_fuse_ignores_unweighted_detector_names():
    results = [DetectorResult("unknown_detector", 1.0, {})]
    fused = _engine().fuse(results)
    assert fused.threat_score == 0.0
    assert fused.contributions == {}


def test_fusion_weights_must_sum_to_one():
    with pytest.raises(ValueError):
        FusionWeights(frequency=0.5, entropy=0.5, statistical=0.5, machine_learning=0.5)
