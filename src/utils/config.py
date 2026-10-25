"""Configuration dataclasses for the AutoDefend pipeline.

All tunable parameters for windowing, the Random Forest model, the fusion
engine, and filesystem paths are centralized here so behaviour can be
changed without touching pipeline logic.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class WindowConfig:
    """Sliding window parameters. Windows are drawn over row order since
    the dataset has no timestamps.
    """

    size: int = 100
    stride: int = 50

    def __post_init__(self) -> None:
        if self.size <= 0:
            raise ValueError("window size must be positive")
        if self.stride <= 0:
            raise ValueError("window stride must be positive")


@dataclass(frozen=True)
class RandomForestConfig:
    """Hyperparameters for the Random Forest classifier."""

    n_estimators: int = 200
    max_depth: int | None = None
    random_state: int = 42
    n_jobs: int = -1


@dataclass(frozen=True)
class SplitConfig:
    """Train/test split settings.

    ``blocked`` is the primary, leakage-safe burst-aware split (see
    ``src/data/split.py``). ``random_legacy`` is the original random split
    over overlapping windows; it LEAKS frames between train and test and
    exists only to quantify how optimistic it was.
    """

    strategy: str = "blocked"
    test_fraction: float = 0.2
    random_state: int = 42
    include_legacy_comparison: bool = False

    def __post_init__(self) -> None:
        if self.strategy not in ("blocked", "random_legacy"):
            raise ValueError("split strategy must be 'blocked' or 'random_legacy'")
        if not 0.0 < self.test_fraction < 1.0:
            raise ValueError("test_fraction must be in (0, 1)")


@dataclass(frozen=True)
class CalibrationConfig:
    """Rule-threshold calibration, computed from normal TRAINING windows only.

    For each rule feature, a detector's confidence ramps from 0 at the
    ``ramp_start_tail`` quantile of normal-training values (e.g. the 99th
    percentile for "high is suspicious") to 1 at the ``ramp_end_tail``
    quantile (the 99.9th). Lower-tail features use the mirrored quantiles.
    """

    ramp_start_tail: float = 0.01
    ramp_end_tail: float = 0.001

    def __post_init__(self) -> None:
        if not 0.0 < self.ramp_end_tail < self.ramp_start_tail < 0.5:
            raise ValueError("require 0 < ramp_end_tail < ramp_start_tail < 0.5")


@dataclass(frozen=True)
class FusionWeights:
    """Relative weight each detector contributes to the fused threat score.

    Weights must sum to 1.0 so the fused score stays within 0-100.
    """

    frequency: float = 0.20
    entropy: float = 0.20
    statistical: float = 0.15
    machine_learning: float = 0.45

    def __post_init__(self) -> None:
        total = self.frequency + self.entropy + self.statistical + self.machine_learning
        if not math.isclose(total, 1.0, abs_tol=1e-6):
            raise ValueError(f"fusion weights must sum to 1.0, got {total}")

    def as_dict(self) -> dict[str, float]:
        return {
            "frequency": self.frequency,
            "entropy": self.entropy,
            "statistical": self.statistical,
            "machine_learning": self.machine_learning,
        }


@dataclass(frozen=True)
class ThreatThresholds:
    """Lower-bound threat scores (0-100) for each threat level."""

    low: float = 20.0
    medium: float = 40.0
    high: float = 60.0
    critical: float = 80.0

    def __post_init__(self) -> None:
        if not (0.0 <= self.low < self.medium < self.high < self.critical <= 100.0):
            raise ValueError("thresholds must satisfy 0 <= low < medium < high < critical <= 100")


@dataclass(frozen=True)
class PathConfig:
    """Filesystem locations used by the pipeline."""

    data_file: Path = Path("data/Car_Hacking.csv")
    model_file: Path = Path("artifacts/models/random_forest.joblib")
    results_dir: Path = Path("artifacts/results")


@dataclass(frozen=True)
class PipelineConfig:
    """Aggregate configuration for a single end-to-end pipeline run."""

    window: WindowConfig = field(default_factory=WindowConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    random_forest: RandomForestConfig = field(default_factory=RandomForestConfig)
    fusion_weights: FusionWeights = field(default_factory=FusionWeights)
    thresholds: ThreatThresholds = field(default_factory=ThreatThresholds)
    paths: PathConfig = field(default_factory=PathConfig)
