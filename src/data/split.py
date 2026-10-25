"""Leakage-safe, burst-aware train/test splitting for the CAN dataset.

Why not a random split over sliding windows? Windows overlap (stride <
size), so a random split puts frames from the same window into both train
and test, and attacks arrive in bursts, so neighbouring windows are
near-copies of each other. That inflates every metric.

Strategy implemented here (``blocked`` split):

1. Find *bursts*: maximal runs of attack frames, where two attack frames
   belong to the same burst unless they are separated by at least
   ``min_gap`` consecutive normal frames (``min_gap`` = window size). A
   window can therefore never span two bursts across a gap.
2. Group consecutive bursts of the same attack label into *attack
   regions*; the traffic between attack regions forms *normal regions*.
3. Inside every region, the chronologically **last** ``test_fraction`` is
   held out as test (for attack regions: the last bursts, with the cut
   placed in the all-normal gap between two bursts, which is dropped).
4. Windows are generated **separately inside each train/test segment**,
   after the split. No window crosses a boundary, so no frame is shared,
   and the nearest train and test frames are at least ``min_gap`` apart.

Nothing is shuffled and no timestamps are invented: row order is the only
time axis the dataset provides. Limitation: the dataset holds one
recording per attack type, so this measures generalization to *unseen
bursts later in the same recording*, not to unseen attack campaigns.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.data.schema import NORMAL_LABEL


@dataclass(frozen=True)
class Burst:
    """A contiguous injection episode: rows ``[start, end)`` of one attack label."""

    index: int
    start: int
    end: int
    label: str


@dataclass(frozen=True)
class Segment:
    """A contiguous row range ``[start, end)`` assigned to train or test."""

    start: int
    end: int
    subset: str  # "train" or "test"
    kind: str  # "attack" or "normal"

    @property
    def length(self) -> int:
        return self.end - self.start


@dataclass(frozen=True)
class BlockedSplit:
    """Result of the burst-aware split."""

    train_segments: tuple[Segment, ...]
    test_segments: tuple[Segment, ...]
    bursts: tuple[Burst, ...]
    dropped_rows: int


def find_bursts(labels: np.ndarray, min_gap: int) -> list[Burst]:
    """Group attack frames into bursts separated by >= ``min_gap`` normal frames."""
    attack_positions = np.flatnonzero(labels != NORMAL_LABEL)
    if attack_positions.size == 0:
        return []

    # diff - 1 is the number of normal frames between consecutive attack frames.
    breaks = np.flatnonzero(np.diff(attack_positions) - 1 >= min_gap)
    first_idx = np.concatenate(([0], breaks + 1))
    last_idx = np.concatenate((breaks, [attack_positions.size - 1]))

    bursts: list[Burst] = []
    for i, (a, b) in enumerate(zip(first_idx, last_idx)):
        start = int(attack_positions[a])
        end = int(attack_positions[b]) + 1
        burst_labels, counts = np.unique(labels[start:end][labels[start:end] != NORMAL_LABEL], return_counts=True)
        bursts.append(Burst(index=i, start=start, end=end, label=str(burst_labels[counts.argmax()])))
    return bursts


def blocked_split(labels: np.ndarray, window_size: int, test_fraction: float) -> BlockedSplit:
    """Split row indices into leakage-safe train/test segments.

    Args:
        labels: Per-frame labels in original row order.
        window_size: Window length; also the minimum normal gap that
            separates two bursts and the buffer dropped at boundaries.
        test_fraction: Fraction of each region (by bursts for attack
            regions, by rows for normal regions) held out as test.

    Returns:
        A BlockedSplit with ordered, non-overlapping segments.
    """
    if not 0.0 < test_fraction < 1.0:
        raise ValueError("test_fraction must be in (0, 1)")

    labels = np.asarray(labels)
    n_rows = len(labels)
    bursts = find_bursts(labels, min_gap=window_size)

    # Group consecutive bursts with the same label into attack regions.
    regions: list[list[Burst]] = []
    for burst in bursts:
        if regions and regions[-1][-1].label == burst.label:
            regions[-1].append(burst)
        else:
            regions.append([burst])

    segments: list[Segment] = []
    cursor = 0
    for region in regions:
        region_start, region_end = region[0].start, region[-1].end
        segments.extend(_split_normal(cursor, region_start, test_fraction))
        segments.extend(_split_attack_region(region, test_fraction))
        cursor = region_end
    segments.extend(_split_normal(cursor, n_rows, test_fraction))

    segments, dropped = _trim_boundaries(segments, window_size)
    train = tuple(s for s in segments if s.subset == "train")
    test = tuple(s for s in segments if s.subset == "test")
    return BlockedSplit(train_segments=train, test_segments=test, bursts=tuple(bursts), dropped_rows=dropped)


def blocked_fold_split(labels: np.ndarray, window_size: int, n_folds: int, fold: int) -> BlockedSplit:
    """Rotating blocked holdout: fold ``fold`` of ``n_folds`` is the test set.

    Every region (attack region of consecutive same-label bursts, or normal
    region) is cut into ``n_folds`` contiguous chronological chunks - by
    burst count for attack regions, by rows for normal regions - and chunk
    ``fold`` is held out. Nothing is shuffled. Across all folds every burst
    is a test burst exactly once. Boundary handling (dropped normal-only
    gaps, >= ``window_size`` frames between train and test, attack frames
    never dropped) is the same as ``blocked_split``; the last fold equals
    ``blocked_split(..., test_fraction=1 / n_folds)``.
    """
    if n_folds < 2 or not 0 <= fold < n_folds:
        raise ValueError("need n_folds >= 2 and 0 <= fold < n_folds")

    labels = np.asarray(labels)
    n_rows = len(labels)
    bursts = find_bursts(labels, min_gap=window_size)

    regions: list[list[Burst]] = []
    for burst in bursts:
        if regions and regions[-1][-1].label == burst.label:
            regions[-1].append(burst)
        else:
            regions.append([burst])

    segments: list[Segment] = []
    cursor = 0
    for region in regions:
        if len(region) < n_folds:
            raise ValueError(f"attack region '{region[0].label}' has {len(region)} bursts, fewer than n_folds={n_folds}")
        segments.extend(_fold_normal(cursor, region[0].start, n_folds, fold))
        segments.extend(_fold_attack_region(region, n_folds, fold))
        cursor = region[-1].end
    segments.extend(_fold_normal(cursor, n_rows, n_folds, fold))

    segments, dropped = _trim_boundaries(segments, window_size)
    train = tuple(s for s in segments if s.subset == "train")
    test = tuple(s for s in segments if s.subset == "test")
    return BlockedSplit(train_segments=train, test_segments=test, bursts=tuple(bursts), dropped_rows=dropped)


def _fold_normal(start: int, end: int, n_folds: int, fold: int) -> list[Segment]:
    length = end - start
    if length <= 0:
        return []
    lo = start + int(round(fold * length / n_folds))
    hi = start + int(round((fold + 1) * length / n_folds))
    parts = [Segment(start, lo, "train", "normal"), Segment(lo, hi, "test", "normal"), Segment(hi, end, "train", "normal")]
    return [p for p in parts if p.length > 0]


def _fold_attack_region(region: list[Burst], n_folds: int, fold: int) -> list[Segment]:
    n = len(region)
    lo = int(round(fold * n / n_folds))
    hi = int(round((fold + 1) * n / n_folds))
    before, held, after = region[:lo], region[lo:hi], region[hi:]
    segments: list[Segment] = []
    for bursts, subset in ((before, "train"), (held, "test"), (after, "train")):
        if bursts:
            segments.append(Segment(bursts[0].start, bursts[-1].end, subset, "attack"))
    return segments


def _split_normal(start: int, end: int, test_fraction: float) -> list[Segment]:
    """Split a normal-only region chronologically: train first, then test."""
    length = end - start
    if length <= 0:
        return []
    cut = start + int(round((1.0 - test_fraction) * length))
    parts = [Segment(start, cut, "train", "normal"), Segment(cut, end, "test", "normal")]
    return [p for p in parts if p.length > 0]


def _split_attack_region(region: list[Burst], test_fraction: float) -> list[Segment]:
    """Hold out the last bursts of an attack region; drop the gap at the cut."""
    n_test = int(round(test_fraction * len(region)))
    if len(region) >= 2:
        n_test = min(max(n_test, 1), len(region) - 1)
    else:
        n_test = 0

    train_bursts, test_bursts = region[: len(region) - n_test], region[len(region) - n_test :]
    segments: list[Segment] = []
    if train_bursts:
        segments.append(Segment(train_bursts[0].start, train_bursts[-1].end, "train", "attack"))
    if test_bursts:
        segments.append(Segment(test_bursts[0].start, test_bursts[-1].end, "test", "attack"))
    return segments


def _trim_boundaries(segments: list[Segment], buffer: int) -> tuple[list[Segment], int]:
    """Drop ``buffer`` all-normal rows where adjacent segments change subset.

    Attack frames are never dropped: the buffer is taken from the normal
    side of any boundary. Boundaries inside an attack region already
    contain a dropped gap of >= ``buffer`` normal frames.
    """
    segments = list(segments)
    dropped = 0
    for i in range(len(segments) - 1):
        a, b = segments[i], segments[i + 1]
        if a.subset == b.subset or a.end != b.start:
            continue
        if a.kind == "normal" and b.kind == "attack":
            trim = min(buffer, a.length)
            segments[i] = Segment(a.start, a.end - trim, a.subset, a.kind)
        else:  # attack|normal, or normal|normal (cut inside a normal region): trim the later segment
            trim = min(buffer, b.length)
            segments[i + 1] = Segment(b.start + trim, b.end, b.subset, b.kind)
        dropped += trim

    kept = [s for s in segments if s.length > 0]
    return kept, dropped
