import numpy as np
import pandas as pd

from src.data.split import blocked_split, find_bursts
from src.data.windowing import iter_segment_windows

WINDOW = 10


def _labels() -> np.ndarray:
    """normal | DoS region (10 bursts) | normal | Fuzzy region (10 bursts) | normal."""
    out = ["R"] * 80
    for attack in ("DoS", "Fuzzy"):
        for _ in range(10):
            burst = ["R" if i % 3 == 0 else attack for i in range(1, 21)]  # sparse, interleaved attack frames
            burst[0] = attack
            burst[-1] = attack
            out += burst + ["R"] * 15  # gap >= WINDOW separates bursts
        out += ["R"] * 100
    return np.array(out)


def _windows(split, df, stride=5):
    tr = list(iter_segment_windows(df, [(s.start, s.end) for s in split.train_segments], WINDOW, stride))
    te = list(iter_segment_windows(df, [(s.start, s.end) for s in split.test_segments], WINDOW, stride))
    return tr, te


def test_find_bursts_groups_by_gap_and_labels():
    bursts = find_bursts(_labels(), min_gap=WINDOW)
    assert len(bursts) == 20
    assert [b.label for b in bursts] == ["DoS"] * 10 + ["Fuzzy"] * 10


def test_blocked_split_is_deterministic():
    assert blocked_split(_labels(), WINDOW, 0.2) == blocked_split(_labels(), WINDOW, 0.2)


def test_blocked_split_holds_out_last_bursts_of_every_attack_region():
    labels = _labels()
    split = blocked_split(labels, WINDOW, 0.2)
    train_rows = {i for s in split.train_segments for i in range(s.start, s.end)}
    test_rows = {i for s in split.test_segments for i in range(s.start, s.end)}
    assert train_rows and test_rows
    assert not (train_rows & test_rows)

    for attack in ("DoS", "Fuzzy"):
        region = [b for b in split.bursts if b.label == attack]
        # 20% of 10 bursts -> the last 2 are test; the first 8 are train
        assert all(b.start in train_rows for b in region[:8])
        assert all(b.start in test_rows for b in region[8:])


def test_blocked_split_never_shares_frames_or_bursts_between_train_and_test_windows():
    labels = _labels()
    df = pd.DataFrame({"x": range(len(labels))})
    split = blocked_split(labels, WINDOW, 0.2)
    tr, te = _windows(split, df)
    assert tr and te

    train_frames = {i for start, _ in tr for i in range(start, start + WINDOW)}
    test_frames = {i for start, _ in te for i in range(start, start + WINDOW)}
    assert not (train_frames & test_frames)

    for burst in split.bursts:
        in_train = any(start < burst.end and start + WINDOW > burst.start for start, _ in tr)
        in_test = any(start < burst.end and start + WINDOW > burst.start for start, _ in te)
        assert not (in_train and in_test), f"burst {burst.index} appears in both train and test"


def test_blocked_split_keeps_a_gap_between_train_and_test_frames():
    split = blocked_split(_labels(), WINDOW, 0.2)
    train_rows = np.array(sorted(i for s in split.train_segments for i in range(s.start, s.end)))
    for s in split.test_segments:
        for edge in (s.start, s.end - 1):
            assert np.abs(train_rows - edge).min() >= WINDOW


def test_blocked_split_never_drops_attack_frames():
    labels = _labels()
    split = blocked_split(labels, WINDOW, 0.2)
    kept = {i for s in (*split.train_segments, *split.test_segments) for i in range(s.start, s.end)}
    assert all(i in kept for i in np.flatnonzero(labels != "R"))


# ---- repeated blocked folds -------------------------------------------------
import pytest  # noqa: E402

from src.data.split import blocked_fold_split  # noqa: E402

N_FOLDS = 5


def _fold_splits():
    return [blocked_fold_split(_labels(), WINDOW, N_FOLDS, k) for k in range(N_FOLDS)]


def _test_burst_ids(split):
    test_rows = {i for s in split.test_segments for i in range(s.start, s.end)}
    return {b.index for b in split.bursts if b.start in test_rows}


def test_exactly_five_folds_and_invalid_fold_rejected():
    assert len(_fold_splits()) == 5
    with pytest.raises(ValueError):
        blocked_fold_split(_labels(), WINDOW, N_FOLDS, 5)


def test_each_fold_holds_out_the_expected_number_of_bursts_per_type():
    for split in _fold_splits():
        test_ids = _test_burst_ids(split)
        for attack in ("DoS", "Fuzzy"):
            assert len([b for b in split.bursts if b.label == attack and b.index in test_ids]) == 2  # 10 bursts / 5 folds


def test_folds_are_deterministic():
    assert _fold_splits() == _fold_splits()


def test_folds_cover_every_burst_exactly_once_in_chronological_blocks():
    per_fold = [_test_burst_ids(s) for s in _fold_splits()]
    all_ids = [i for ids in per_fold for i in ids]
    assert sorted(all_ids) == list(range(20))  # every burst is test exactly once
    # fold k holds out the k-th contiguous block of each type (bursts 0-9 DoS, 10-19 Fuzzy)
    assert per_fold[0] == {0, 1, 10, 11}
    assert per_fold[4] == {8, 9, 18, 19}


def test_no_burst_or_frame_on_both_sides_in_any_fold():
    labels = _labels()
    df = pd.DataFrame({"x": range(len(labels))})
    for split in _fold_splits():
        tr, te = _windows(split, df)
        assert tr and te
        assert not ({i for s, _ in tr for i in range(s, s + WINDOW)} & {i for s, _ in te for i in range(s, s + WINDOW)})
        for burst in split.bursts:
            in_train = any(s < burst.end and s + WINDOW > burst.start for s, _ in tr)
            in_test = any(s < burst.end and s + WINDOW > burst.start for s, _ in te)
            assert not (in_train and in_test)


def test_last_fold_reproduces_the_single_blocked_split():
    assert blocked_fold_split(_labels(), WINDOW, N_FOLDS, 4) == blocked_split(_labels(), WINDOW, 0.2)


def test_fold_splits_never_drop_attack_frames():
    labels = _labels()
    for split in _fold_splits():
        kept = {i for s in (*split.train_segments, *split.test_segments) for i in range(s.start, s.end)}
        assert all(i in kept for i in np.flatnonzero(labels != "R"))
