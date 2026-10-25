"""Data-driven calibration of the rule-based detectors' thresholds.

The rule agents turn a feature into a 0-1 confidence with a linear ramp
(``scale_confidence``). This module derives the ramp endpoints from the
**normal TRAINING windows only**:

* "high is suspicious" features: ramp from the ``1 - ramp_start_tail``
  quantile (default 99th percentile) to the ``1 - ramp_end_tail``
  quantile (default 99.9th) of normal-training values.
* "low is suspicious" features: ramp from the ``ramp_start_tail`` quantile
  (1st percentile) down to the ``ramp_end_tail`` quantile (0.1th).

Only the feature matrix of normal training windows is passed in, so test
data cannot influence the thresholds, and the computation is deterministic
(quantiles, no randomness). Attack labels are not used at all.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from src.features.schema import WindowFeatures
from src.utils.config import CalibrationConfig

# Two ramps of a degenerate (equal-quantile) feature collapse to a step.
_MIN_RAMP_WIDTH = 1e-9


@dataclass(frozen=True)
class CalibratedThresholds:
    """Constructor kwargs for each rule agent, plus provenance."""

    frequency: dict[str, float]
    entropy: dict[str, float]
    statistical: dict[str, float]
    n_normal_training_windows: int
    config: CalibrationConfig

    def to_dict(self) -> dict[str, Any]:
        return {
            "frequency": self.frequency,
            "entropy": self.entropy,
            "statistical": self.statistical,
            "n_normal_training_windows": self.n_normal_training_windows,
            "ramp_start_tail": self.config.ramp_start_tail,
            "ramp_end_tail": self.config.ramp_end_tail,
        }


def calibrate_rule_thresholds(
    normal_training_features: np.ndarray,
    config: CalibrationConfig | None = None,
) -> CalibratedThresholds:
    """Derive rule-agent thresholds from normal training windows.

    Args:
        normal_training_features: Matrix (n_windows, n_features) of
            ``WindowFeatures.to_array()`` rows for windows labeled normal
            in the TRAINING set. Columns follow ``WindowFeatures.feature_names()``.
        config: Quantile settings.

    Returns:
        Thresholds in the keyword format expected by the rule agents.
    """
    config = config or CalibrationConfig()
    if normal_training_features.ndim != 2 or normal_training_features.shape[0] == 0:
        raise ValueError("need a non-empty 2-D matrix of normal training windows")

    columns = {name: normal_training_features[:, i] for i, name in enumerate(WindowFeatures.feature_names())}
    s, e = config.ramp_start_tail, config.ramp_end_tail

    def high(name: str) -> tuple[float, float]:
        start = float(np.quantile(columns[name], 1.0 - s))
        end = float(np.quantile(columns[name], 1.0 - e))
        return start, max(end, start + _MIN_RAMP_WIDTH)

    def low(name: str) -> tuple[float, float]:
        start = float(np.quantile(columns[name], s))
        end = float(np.quantile(columns[name], e))
        return start, min(end, start - _MIN_RAMP_WIDTH)

    dominant_low, dominant_high = high("dominant_id_ratio")
    burst_low, burst_high = high("max_burst_ratio")
    id_ceiling, id_floor = low("id_entropy")
    pay_low_ceiling, pay_low_floor = low("payload_entropy")
    pay_high_floor, pay_high_ceiling = high("payload_entropy")
    dup_low, dup_high = high("duplicate_ratio")
    var_flood_ceiling, var_flood_floor = low("payload_variance")
    var_high_floor, var_high_ceiling = high("payload_variance")

    return CalibratedThresholds(
        frequency={
            "dominant_id_low": dominant_low,
            "dominant_id_high": dominant_high,
            "burst_low": burst_low,
            "burst_high": burst_high,
        },
        entropy={
            "id_entropy_ceiling": id_ceiling,
            "id_entropy_floor": id_floor,
            "payload_entropy_low_ceiling": pay_low_ceiling,
            "payload_entropy_low_floor": pay_low_floor,
            "payload_entropy_high_floor": pay_high_floor,
            "payload_entropy_high_ceiling": pay_high_ceiling,
        },
        statistical={
            "duplicate_low": dup_low,
            "duplicate_high": dup_high,
            "variance_flood_ceiling": var_flood_ceiling,
            "variance_flood_floor": var_flood_floor,
            "variance_high_floor": var_high_floor,
            "variance_high_ceiling": var_high_ceiling,
        },
        n_normal_training_windows=int(normal_training_features.shape[0]),
        config=config,
    )
