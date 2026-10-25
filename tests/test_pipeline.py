import numpy as np

from src.pipeline import WindowSet, _run_experiment, summarize_split
from src.utils.config import PipelineConfig, RandomForestConfig
from tests.test_ablation import _features


def _window_set(seed: int, n: int, shift: float = 1.0) -> WindowSet:
    rng = np.random.default_rng(seed)
    labels = np.array(["R"] * (n // 2) + ["DoS"] * (n // 2))
    feats = [_features(rng, lab != "R") for lab in labels]
    starts = np.arange(n) * 100
    return WindowSet(starts, feats, labels)


def _config() -> PipelineConfig:
    return PipelineConfig(random_forest=RandomForestConfig(n_estimators=10))


def test_experiment_thresholds_are_independent_of_test_windows():
    train = _window_set(0, 200)
    summary = {"strategy": "unit", "n_train_windows": 200, "n_test_windows": 100}
    a = _run_experiment(train, _window_set(1, 100), summary, _config())
    # Replace the test set with wildly different data: calibration must not move.
    weird = _window_set(2, 100)
    weird = WindowSet(weird.starts, [_features(np.random.default_rng(9), True)] * 100, weird.labels)
    b = _run_experiment(train, weird, summary, _config())
    assert a.thresholds.to_dict() == b.thresholds.to_dict()
    assert a.ablation.n_test_windows == b.ablation.n_test_windows == 100


def test_experiment_calibrates_on_normal_training_windows_only():
    train = _window_set(0, 200)
    result = _run_experiment(train, _window_set(1, 100), {}, _config())
    assert result.thresholds.n_normal_training_windows == int(np.sum(train.labels == "R")) == 100


def test_summarize_split_detects_shared_frames():
    size = 100
    train = WindowSet(np.array([0, 50]), [None, None], np.array(["R", "R"]))
    leaky_test = WindowSet(np.array([100]), [None], np.array(["R"]))  # overlaps train window starting at 50
    clean_test = WindowSet(np.array([400]), [None], np.array(["R"]))
    assert summarize_split("x", train, leaky_test, size, n_rows=600)["shared_frames_between_train_and_test"] == 50
    clean = summarize_split("x", train, clean_test, size, n_rows=600)
    assert clean["shared_frames_between_train_and_test"] == 0
    assert clean["min_frame_distance_train_to_test"] == 250


def test_assert_no_leakage_fails_loudly():
    import pytest

    from src.evaluation.blocked_cv import LeakageError, assert_no_leakage

    ok = {"shared_frames_between_train_and_test": 0, "bursts_shared_between_train_and_test": 0, "min_frame_distance_train_to_test": 120}
    assert_no_leakage(ok, 100)
    for bad in (
        {**ok, "shared_frames_between_train_and_test": 1},
        {**ok, "bursts_shared_between_train_and_test": 1},
        {**ok, "min_frame_distance_train_to_test": 50},
    ):
        with pytest.raises(LeakageError):
            assert_no_leakage(bad, 100)
