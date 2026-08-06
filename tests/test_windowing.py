import pandas as pd

from src.data.windowing import iter_window_bounds, iter_windows, label_window


def test_iter_window_bounds_covers_full_range_without_overlap_gap():
    bounds = list(iter_window_bounds(n_rows=10, window_size=4, stride=4))
    assert bounds == [(0, 4), (4, 8)]


def test_iter_window_bounds_supports_overlapping_stride():
    bounds = list(iter_window_bounds(n_rows=10, window_size=4, stride=2))
    assert bounds == [(0, 4), (2, 6), (4, 8), (6, 10)]


def test_iter_window_bounds_drops_final_partial_window():
    bounds = list(iter_window_bounds(n_rows=9, window_size=4, stride=4))
    assert bounds == [(0, 4), (4, 8)]


def test_iter_window_bounds_empty_when_window_exceeds_rows():
    assert list(iter_window_bounds(n_rows=3, window_size=10, stride=1)) == []


def test_iter_windows_yields_expected_slice_length():
    df = pd.DataFrame({"x": range(10)})
    windows = list(iter_windows(df, window_size=5, stride=5))
    assert len(windows) == 2
    assert all(len(w) == 5 for w in windows)


def test_label_window_returns_normal_when_all_rows_normal():
    labels = pd.Series(["R", "R", "R"])
    assert label_window(labels) == "R"


def test_label_window_flags_single_attack_frame_among_normal_traffic():
    labels = pd.Series(["R"] * 99 + ["DoS"])
    assert label_window(labels) == "DoS"


def test_label_window_picks_most_frequent_attack_label():
    labels = pd.Series(["R", "DoS", "DoS", "Fuzzy"])
    assert label_window(labels) == "DoS"
