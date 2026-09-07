# Byzantine-Resilient Consensus for Distributed BMS — Reproducibility Package

Code and experiment scripts for the manuscript

> **Byzantine-Resilient Consensus for Distributed Battery Management:
> separating node faults from battery faults on 70 days of real LiFePO4 field data**

(Applied Energy; previous submission APEN-D-26-09704R1.)

## Dataset

El Tiemblo Solar+Storage LiFePO4 dataset (70 days, 22 524 samples) on Zenodo:
**DOI [10.5281/zenodo.20717244](https://doi.org/10.5281/zenodo.20717244)** (CC-BY-4.0).

The package has **two generations of experiments**. Both are kept, because the
second one exists to answer specific reviewer objections to the first and the
comparison is part of the argument.

| Generation | Scripts | What it is |
|---|---|---|
| **v3 (round 2, current)** | `signal_model.py`, `faults.py`, `abr_detector.py`, `revision2_experiments.py`, `generate_figures_v3.py` | Signal-level fault injection, physically grounded module dispersion, battery-fault taxonomy, physical-consistency router |
| **v1/v2 (round 1, provenance)** | `run_evaluation.py`, `revision_experiments.py`, `run_sensitivity.py`, `generate_figures.py` | Report-level fault injection, module SOC = pack SOC + N(0, 0.005). Kept unmodified so the round-1 numbers remain reproducible |

## Why the experiments were rebuilt

Two reviewers showed that the round-1 validation did not exercise the claim the
paper makes.

* Faults were injected by editing the final SOC number a node reported, so the
  central claim — that a healthy-but-aged module is not mistaken for a
  misbehaving node — was never tested.
* `simulate_modules` set every module's SOC to the pack SOC plus N(0, 0.005),
  making the 16 nodes near-copies of one another.
* Nothing in the round-1 taxonomy said anything about internal short circuit or
  cell imbalance.
* Table 11 showed the consensus RMSE identical to five decimals across four
  interpolation schemes, i.e. the per-module detail did not contribute.

The v3 package addresses each of these. It also fixes two defects found while
rebuilding, both documented in `REVISION2_RESULTS.md`: the round-1 code
integrated current with the **wrong sign** for this dataset, and it kept the
full-rate `dt` after subsampling, under-counting charge throughput by ~2x.

## Contents

| File | Purpose |
|---|---|
| `config_v3.json` | Every parameter: signal model, fault classes, detector, router thresholds, seeds |
| `signal_model.py` | Per-module signal reconstruction anchored to the measured cell-voltage envelope + honest local estimator |
| `faults.py` | 16 fault classes in three families (node / glitch / battery), injected at signal or message level |
| `abr_detector.py` | Statistical detector, physical-consistency router, message-freshness (`comms-degraded`) handling, and all baselines |
| `revision2_experiments.py` | E7–E12 and E1'/E2'/E6' |
| `generate_figures_v3.py` | figR6–figR9, figR1v3, figR2v3, figR5v3 |
| `generate_graphical_abstract_v3.py` | Graphical abstract (numbers read from the results JSON) |
| `results/revision2_results.json` | Round-2 results, iteration 1 (current): mean ± std over 5 seeds plus per-seed values |
| `results/revision2_results_iter0.json` | Round-2 results, iteration 0, preserved for the before/after comparison |
| `results/aggregated_results.json` | Consolidated; round-2 results live under the `v3` key |
| `REVISION2_RESULTS.md` | Round-2 findings, including the unfavourable ones |
| `REVISION_RESULTS.md` | Round-1 findings (provenance) |

## Requirements

- Python 3.11+ (developed on 3.13.7)
- `pip install -r requirements.txt` (numpy, pandas, scipy, matplotlib, scikit-learn)

```bash
python -m venv .venv
.venv/Scripts/activate          # Windows;  source .venv/bin/activate elsewhere
pip install -r requirements.txt
```

## Data

The El Tiemblo 70-day LiFePO4 solar-storage dataset (22 524 samples, 16s16p
pack) is released separately with a persistent DOI; see the manuscript's Data
Availability statement. Place the CSV at

```
<repo-parent>/datasets/eltiemblo_solar_completo/848299_0_Hock_log_20250101-0000_to_20251231-2358.csv
```

or edit `DATA_FILE` in `signal_model.py`. The round-1 cross-validation set is
the NASA PCoE Li-ion aging dataset (cells B5/B6/B7), saved as
`datasets/Battery_dataset.csv`; it is used only by `revision_experiments.py`.

**Current sign convention.** Positive current is charging. This was verified
against the data rather than assumed: mean `I_pack` is +175.7 A over samples
where the BMS SOC rises and −56.2 A where it falls, and the windowed regression
of ΔSOC on ∫I dt has a positive slope. `run_evaluation.py` (round 1) integrated
with the opposite sign.

## Reproducing the round-2 results

```bash
python revision2_experiments.py          # E7-E11, E1', E2', E6'  (~30 min, CPU only)
python generate_figures_v3.py            # all round-2 figures
python generate_graphical_abstract_v3.py # graphical abstract
```

`revision2_experiments.py` first calibrates the router thresholds on
**fault-free** runs over the first 10 % of the horizon, writes them back into
`config_v3.json` under `detector.router`, and then freezes them. The thresholds
are never set by looking at the fault classes.

It then runs **E12**, the CUSUM-persistence sweep, and picks the detector
configuration used by the rest of the suite with a criterion declared in
advance in `config_v3.json`: false-positive rate exactly zero on the fault-free
control and zero quarantines on the eight classes that must not be quarantined,
and among the configurations that pass, the highest mean probability of
quarantining the target across the eight node-fault classes. On the current
data this selects `gap-first, P=1` — i.e. persistence is rejected. See
`REVISION2_RESULTS.md`, "Iteration 1", for why, and for the separation between
what message freshness contributes and what the physical router contributes.

Every stochastic experiment runs over the 5 fixed seeds 42, 123, 456, 789, 1024;
reported metrics are mean ± std over those seeds, and the per-seed values are
stored in the JSON.

### Measured runtime

`revision2_experiments.py`: **see `_meta.runtime_seconds` in
`results/revision2_results.json`** — 1 696 s (28.3 min) for iteration 1 on a
CPU-only desktop with single-threaded NumPy; iteration 0 took 2 226 s
(37.1 min) before E12 was added and before the detector loop was segmented.
Figures add roughly one minute, the graphical abstract a few seconds.

## Reproducing the round-1 results (provenance)

```bash
python revision_experiments.py
python run_evaluation.py
python run_sensitivity.py
python generate_figures.py
```

## License

MIT — see [LICENSE](LICENSE).
