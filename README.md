# AutoDefend

A modular Intrusion Detection System (IDS) for automotive Controller Area
Networks (CAN). AutoDefend detects malicious CAN traffic in the
[HCRL Car-Hacking dataset](http://ocslab.hksecurity.net/Datasets/car-hacking-dataset)
by combining lightweight statistical/rule-based detectors with a Random
Forest classifier, fused into a single 0-100 threat score.

This is a portfolio project. Its goal is to demonstrate software
engineering, data engineering, and applied ML practice, not to chase a
leaderboard accuracy number.

## Overview

Modern vehicles broadcast hundreds of CAN messages per second over a bus
with no built-in authentication — any device with physical or remote
access can inject frames. AutoDefend ingests a stream of CAN frames,
groups them into fixed-size sliding windows (there are no timestamps in
this dataset, so windows are drawn over message order), extracts
statistical features per window, scores each window with four
independent detectors, and fuses their outputs into a single threat
verdict.

```
Load → Validate → Preprocess → Sliding Windows → Feature Extraction
     → Detection Agents (Frequency / Entropy / Statistical / ML)
     → Fusion Engine → Evaluation → Results
```

## Motivation

Off-the-shelf CAN IDS demos are usually a single notebook that trains one
classifier and reports accuracy. AutoDefend is built as a small
production system instead: a typed, tested, config-driven pipeline with
clear module boundaries, so any single piece (a new detector, a
different fusion strategy, a different model) can be swapped without
touching the rest.

## A note on the labeling scheme

The dataset's attack frames are not injected in long contiguous blocks —
they are single malicious frames interspersed with normal traffic (a
window in the DoS-attack region of the file is still mostly made of
normal "R" rows). Labeling a window by majority vote would therefore
label almost every window "normal," even ones actively under attack.
AutoDefend labels a window with an attack type if **any** frame in it
carries that label — the safety-appropriate choice for an IDS, and the
one that makes the per-window statistics actually mean something. See
[`src/data/windowing.py`](src/data/windowing.py) for the implementation
and rationale.

This same analysis also drove threshold calibration for the rule-based
agents: thresholds are set from percentiles of the normal-traffic
baseline (window_size=100, stride=50) rather than guessed constants —
see the docstrings in [`src/agents/`](src/agents/).

## Architecture

```
src/
    data/         Loading, schema validation, label preprocessing, sliding windows
    features/     Frequency, entropy, and statistical feature extraction
    agents/       Independent detection modules (frequency, entropy, statistical, ML)
    models/       Random Forest classifier (train/predict/save/load)
    fusion/       Combines detector outputs into a threat score + level
    evaluation/   Metrics, classification reports, confusion matrix plots
    utils/        Config dataclasses, logging setup
    pipeline.py   Orchestrates the end-to-end run
main.py           CLI entry point
```

Design principles:

- **Low coupling.** Each detection agent only depends on the shared
  `WindowFeatures` vector, not on each other or on the model. Agents can
  be added, removed, or re-weighted without touching the fusion engine.
- **Config-driven behavior.** Window size/stride, Random Forest
  hyperparameters, fusion weights, and threat-level thresholds are all
  dataclasses in `src/utils/config.py`, not hardcoded constants.
- **Explainability alongside prediction.** Every `DetectorResult` carries
  a `details` dict of the raw values that drove its confidence score, so
  a flagged window can be inspected, not just trusted.

### Detection agents

| Agent | Signal | Catches |
|---|---|---|
| `FrequencyAgent` | Dominant CAN ID ratio, burst ratio | DoS / Gear / RPM flooding |
| `EntropyAgent` | ID & payload Shannon entropy (both directions) | Flooding (depressed entropy) and fuzzing (elevated entropy) |
| `StatisticalAgent` | Duplicate-frame ratio, payload variance | Replayed/near-identical frames |
| `MLAgent` | Random Forest probability mass on non-normal classes | Multi-feature interactions the rule-based agents miss |

### Fusion engine

Each agent contributes `weight × confidence × 100` to a 0-100 threat
score (`src/fusion/engine.py`). Weights are configurable
(`FusionWeights`, default: frequency 0.20, entropy 0.20, statistical
0.15, ML 0.45 — the ML agent is weighted highest because it's the only
detector that learns from full multi-feature interactions rather than a
single threshold rule). The score is bucketed into a `ThreatLevel`:

| Score | Level |
|---|---|
| < 20 | Normal |
| 20-39 | Low |
| 40-59 | Medium |
| 60-79 | High |
| ≥ 80 | Critical |

Adding a new detector only requires implementing `DetectorAgent.analyze`
and registering a weight for its name — `FusionEngine.fuse` needs no
changes.

## Folder structure

```
AutoDefend/
├── data/                       Car_Hacking.csv (gitignored, not tracked)
├── artifacts/                  Generated model + evaluation output (gitignored)
│   ├── models/random_forest.joblib
│   └── results/{evaluation_report.json, confusion_matrix_*.png}
├── src/                        Pipeline source (see Architecture above)
├── tests/                      pytest unit tests
├── main.py                     CLI entry point
├── requirements.txt            Runtime dependencies
├── requirements-dev.txt        + pytest
└── pytest.ini
```

## Installation

Requires Python 3.11+.

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

Place the dataset at `data/Car_Hacking.csv` (columns: `CAN ID`,
`DATA[0]`..`DATA[7]`, `Label`).

## Usage

Run the full pipeline with default settings (100-message windows,
50-message stride, 200-tree Random Forest):

```bash
python main.py
```

This loads and validates the dataset, builds sliding windows, extracts
features, trains and saves the Random Forest, runs all four detection
agents through the fusion engine, and writes evaluation results to
`artifacts/`.

Key options:

```bash
python main.py \
    --data-path data/Car_Hacking.csv \
    --window-size 100 \
    --stride 50 \
    --n-estimators 200 \
    --test-size 0.2 \
    --model-path artifacts/models/random_forest.joblib \
    --results-dir artifacts/results \
    --log-level INFO
```

Run `python main.py --help` for the full list.

### Tests

```bash
pip install -r requirements-dev.txt
pytest
```

### Results (default configuration)

On the full ~818K-row dataset (100-message windows, 50-message stride,
16,367 windows total, 80/20 train/test split):

**Random Forest (multi-class: R / DoS / Fuzzy / Gear / RPM)**

| Metric | Value |
|---|---|
| Accuracy | 0.931 |
| Macro Precision | 0.951 |
| Macro Recall | 0.905 |
| Macro F1 | 0.926 |

**Fusion Engine (binary: Normal vs. Attack)**

| Metric | Value |
|---|---|
| Accuracy | 0.939 |
| Precision | 0.940 |
| Recall | 0.941 |
| F1 | 0.939 |

Full classification reports, confusion matrices (PNG), and feature
importances are written to `artifacts/results/` on every run.

## Future Improvements

- Time-decayed / EWMA features across windows to capture drift, once a
  timestamped dataset is available.
- Per-attack-type rule agents (the current agents are tuned as
  binary normal-vs-attack detectors; splitting flood detection from
  fuzz detection into separate agents would sharpen `details` reporting).
- Online/streaming inference mode instead of batch evaluation, so the
  pipeline can score a live CAN bus feed frame-by-frame.
- Hyperparameter search (grid/Bayesian) for the Random Forest instead of
  fixed defaults.
- A small FastAPI service exposing `/score` for integration testing with
  other tools.
