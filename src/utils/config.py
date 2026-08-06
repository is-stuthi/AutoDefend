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
    test_size: float = 0.2
    n_jobs: int = -1


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
    random_forest: RandomForestConfig = field(default_factory=RandomForestConfig)
    fusion_weights: FusionWeights = field(default_factory=FusionWeights)
    thresholds: ThreatThresholds = field(default_factory=ThreatThresholds)
    paths: PathConfig = field(default_factory=PathConfig)
