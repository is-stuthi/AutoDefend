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
features, splits train/test with the burst-aware blocked split, calibrates the rule
thresholds on training windows, trains and saves the Random Forest, runs the
RF-only / rules-only / fusion ablation on the held-out windows, and writes
results to `artifacts/`.

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

## Evaluation methodology

**Setup.** 818,440 CAN frames; windows of **100 frames, stride 50**; **11 features** per window
(5 frequency, 2 entropy, 4 payload-statistics); window label = most frequent attack label in the
window, or `R` if none. Random Forest: 200 trees, `random_state=42`, sklearn defaults otherwise.

**Why not a random split over windows?** With stride 50 and size 100, consecutive windows share
half their frames, so a random window split puts the same frames in both train and test. The
original evaluation did exactly that (the legacy run below shares 262,600 frames between train
and test). It is kept only as a clearly labeled LEAKY comparison
(`--include-legacy-comparison`).

**Burst-aware blocked split (primary)** - `src/data/split.py`:

1. Find *bursts*: runs of attack frames separated by fewer than 100 consecutive normal frames. The
   data contains exactly 1,200 bursts (300 per attack type).
2. Group consecutive bursts of the same label into attack regions; the traffic between them forms
   normal regions.
3. In every region the chronologically **last 20%** is test (for attack regions: the last 60 of 300
   bursts; the cut falls in the all-normal gap between two bursts and that gap is dropped).
4. Windows are generated **inside** each train/test segment after the split, so no window crosses
   the boundary.

Nothing is shuffled and no timestamps are invented (the dataset has none; row order is the only
time axis). Verified on the real run: 13,081 train / 3,220 test windows, **0 shared frames**,
**0 bursts touched by both sides** (960 train / 240 test), nearest train/test frames are 104 frames
apart.

**Rule calibration** - `src/agents/calibration.py`: after the split, each rule detector's ramp
endpoints are the 99th and 99.9th percentile (for "high is suspicious" features) or the 1st and
0.1st percentile (for "low is suspicious" features) of the **normal training windows only**.
Attack labels and test data are never used; it is deterministic. The thresholds used are saved to
`artifacts/results/calibrated_thresholds.json`. The detectors' constructor defaults are the old
hand-set values and are used only in unit tests.

**Ablation** - `src/evaluation/ablation.py`, same test windows for every method, attack = positive class:

| Method | Decision rule |
|---|---|
| RF-only | RF 5-class prediction; any non-`R` class = attack |
| Rules-only | frequency + entropy + statistical agents through the same `FusionEngine` (weights renormalized to 0.364 / 0.364 / 0.273, no ML agent); attack if fused score >= 20 |
| Fusion | all four agents, default weights (0.20 / 0.20 / 0.15 / 0.45); attack if fused score >= 20 |

No weight or threshold was tuned on the test set; fusion weights and the score threshold are the
original hand-set defaults.

## Results

All numbers below come from `python main.py --include-legacy-comparison` on the full dataset
(`artifacts/results/evaluation_report.json`).

**Primary (leakage-safe blocked split), 3,220 test windows** (R 1543, DoS 332, Fuzzy 306, Gear 496, RPM 543):

| Method | Accuracy | Precision | Recall | F1 | FPR | FNR |
|---|---|---|---|---|---|---|
| RF-only | 0.9460 | 0.9808 | 0.9141 | 0.9463 | 0.0194 | 0.0859 |
| Rules-only | 0.9332 | 0.9747 | 0.8951 | 0.9332 | 0.0253 | 0.1049 |
| Fusion | 0.9401 | 0.9502 | 0.9338 | 0.9420 | 0.0531 | 0.0662 |

RF 5-class on the same test windows: accuracy 0.9317, macro precision 0.9468, macro recall
0.9098, macro F1 0.9269 (per-class recall: DoS 0.89, Fuzzy 0.94, Gear 0.83, R 0.98, RPM 0.90).

**Reading the ablation honestly.** Fusion does *not* beat RF-only on accuracy or F1 (0.9401 vs 0.9460
accuracy, 0.9420 vs 0.9463 F1). It trades precision for recall: it catches more attacks (FNR 6.6%
vs 8.6%) at the cost of more than 2.7x the false-positive rate (5.3% vs 1.9%). Whether that is
a good trade depends on the cost of a missed attack versus a false alarm, and it is a consequence of
the untuned score threshold of 20. The rules alone, with thresholds calibrated on normal training
data and no ML, reach F1 0.933.

**Legacy random split (LEAKY - do not cite)**, 3,274 test windows, same code and calibration:
RF 5-class accuracy 0.9297; binary accuracy / F1 - RF-only 0.9426 / 0.9424, rules-only 0.9227 / 0.9222,
fusion 0.9356 / 0.9364.

**Old vs. new.** The headline numbers barely move (RF 5-class 0.930 vs 0.932; the originally
reported 0.931 / 0.939 came from hand-set thresholds, which this repo no longer uses). So
frame-sharing leakage was *not* what drove the ~93% figure for this feature set: bursts within a
recording look statistically alike, so a model trained on early bursts transfers to later bursts in
the same recording. What changed is that the number is now defensible: no shared frames, no shared
bursts, thresholds fitted on training data only, and a baseline comparison.

## Limitations

- The dataset has one recording per attack type, so the split measures generalization to unseen
  bursts later in the same recording, not to unseen attack campaigns, vehicles or attack parameters.
- Evaluation is window-level (any attack frame labels the window), not frame-level, so it is not
  comparable to frame-level numbers in the literature.
- Test windows overlap each other (stride 50), so the 3,220 test windows are not independent samples;
  there is no confidence interval or repeated-run variance estimate yet.
- Fusion weights and the score threshold are hand-set, not learned or tuned; the threshold of 20 makes
  fusion trade a higher false-positive rate for recall.
- Batch evaluation only. This is not a real-time or streaming detector, and detection latency
  was not measured. The "agents" are rule-based detector modules, not autonomous agents.
- Random Forest hyperparameters are untuned defaults.

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
