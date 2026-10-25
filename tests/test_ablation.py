import numpy as np
import pytest

from src.agents.calibration import calibrate_rule_thresholds
from src.evaluation.ablation import METHODS, rules_only_weights, run_ablation
from src.evaluation.metrics import binary_metrics
from src.features.schema import WindowFeatures
from src.models.random_forest import RandomForestDetector
from src.utils.config import FusionWeights, ThreatThresholds

NAMES = WindowFeatures.feature_names()


def _features(rng, attack: bool) -> WindowFeatures:
    base = dict(
        unique_id_count=20.0, dominant_id_ratio=0.10, repetition_ratio=0.79, id_count_std=2.0,
        max_burst_ratio=0.02, id_entropy=4.2, payload_entropy=4.2, duplicate_ratio=0.35,
        payload_mean=50.0, payload_std=75.0, payload_variance=5600.0,
    )
    for k in base:
        base[k] *= 1 + rng.normal(0, 0.02)
    if attack:
        base["dominant_id_ratio"] = 0.6
        base["id_entropy"] = 2.0
    return WindowFeatures(**base)


def _fixture():
    rng = np.random.default_rng(0)
    labels = np.array(["R"] * 150 + ["DoS"] * 150)
    return [_features(rng, lab != "R") for lab in labels], labels


def _fitted():
    feats, labels = _fixture()
    X = np.vstack([f.to_array() for f in feats])
    model = RandomForestDetector(n_estimators=20, random_state=0)
    model.train(X, labels, NAMES)
    return model, calibrate_rule_thresholds(X[labels == "R"]), feats, labels, X


def test_binary_metrics_known_values():
    truth = [True, True, True, False, False, False, False, False]
    pred = [True, True, False, True, False, False, False, False]
    m = binary_metrics(truth, pred)
    assert (m.tp, m.fn, m.fp, m.tn) == (2, 1, 1, 4)
    assert m.accuracy == pytest.approx(6 / 8)
    assert m.precision == pytest.approx(2 / 3)
    assert m.recall == pytest.approx(2 / 3)
    assert m.false_positive_rate == pytest.approx(1 / 5)
    assert m.false_negative_rate == pytest.approx(1 / 3)
    assert m.normal_recall == pytest.approx(4 / 5)


def test_rules_only_weights_sum_to_one_and_drop_ml():
    w = rules_only_weights(FusionWeights())
    assert w.machine_learning == 0.0
    assert w.frequency + w.entropy + w.statistical == pytest.approx(1.0)


def test_all_methods_score_exactly_the_same_test_windows():
    model, thresholds, feats, labels, _ = _fitted()
    result = run_ablation(model, thresholds, feats, labels, FusionWeights(), ThreatThresholds())

    assert result.n_test_windows == len(labels)
    assert set(result.predictions) == set(METHODS)
    for method in METHODS:
        assert len(result.predictions[method]) == len(labels)
        m = result.metrics[method]
        assert m.tp + m.fn == int(np.sum(labels != "R"))
        assert m.tn + m.fp == int(np.sum(labels == "R"))


def test_ablation_metrics_match_recomputed_values_and_rf_only_matches_multiclass_prediction():
    model, thresholds, feats, labels, X = _fitted()
    result = run_ablation(model, thresholds, feats, labels, FusionWeights(), ThreatThresholds())
    for method in METHODS:
        assert result.metrics[method] == binary_metrics(labels != "R", result.predictions[method])
    assert np.array_equal(result.predictions["rf_only"], np.asarray(model.predict(X)) != "R")


def test_ml_agent_batch_matches_per_window_scoring():
    from src.agents.ml_agent import MLAgent

    model, _, feats, _, _ = _fitted()
    agent = MLAgent(model)
    batch = agent.analyze_batch(feats)
    single = [agent.analyze(f) for f in feats]
    assert [b.confidence for b in batch] == [s.confidence for s in single]
    assert [b.details for b in batch] == [s.details for s in single]
    # and equals what a raw one-row predict_proba call gives
    row = model.predict_proba(feats[3].to_array().reshape(1, -1))[0]
    classes = list(model.classes_)
    assert batch[3].confidence == pytest.approx(1.0 - row[classes.index("R")], abs=1e-12)
