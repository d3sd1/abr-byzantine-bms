## Dataset

El Tiemblo Solar+Storage LiFePO4 dataset (70 days, 22,524 samples) on Zenodo:
**DOI [10.5281/zenodo.20717244](https://doi.org/10.5281/zenodo.20717244)** (CC-BY-4.0).

# ABR Byzantine-Resilient BMS — Reproducibility Package

Code and experiment scripts for the paper:

> **Trust-Weighted Byzantine-Resilient Consensus for Distributed Battery
> Management Systems: A Statistical Fault Detector Validated on 70 Days of
> Real LiFePO4 Solar Data**

This repository reproduces all experimental results (detector ablation,
robustness/threshold sweep, interpolation sensitivity, PINN-vs-filter
comparison, public-dataset cross-validation, and centralised-vs-distributed
evaluation) and regenerates every figure programmatically.

## Contents

| File | Purpose |
|------|---------|
| `revision_experiments.py` | Experiments E1–E6 (ablation, robustness, interpolation, PINN-vs-filter, NASA PCoE cross-validation, centralised vs distributed) |
| `run_evaluation.py` | Main detector / consensus evaluation on the El Tiemblo dataset |
| `run_sensitivity.py` | Hyperparameter sensitivity sweeps (lambda_phys, w1, gap threshold) |
| `generate_figures.py` | Regenerates figR1–figR5 (PDF) from `results/` |
| `generate_graphical_abstract.py` | Graphical abstract |
| `requirements.txt` | Pinned Python dependencies |
| `results/` | Aggregated results (JSON), 5 seeds: 42, 123, 456, 789, 1024 |
| `figures/` | Generated figures (PDF) |

## Requirements

- Python 3.11+
- See `requirements.txt` (numpy, pandas, scipy, matplotlib, scikit-learn)

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Data

The El Tiemblo 70-day LiFePO4 solar-storage dataset (22,524 samples,
16 modules) is released separately on Zenodo with a persistent DOI (see the
paper's Data Availability statement). Place the dataset CSV under a top-level
`datasets/` directory as referenced in `run_evaluation.py` and
`revision_experiments.py`.

The public cross-validation dataset is the NASA Prognostics Center of
Excellence (PCoE) Li-ion battery aging dataset (cells B5/B6/B7), publicly
available from the NASA PCoE data repository. Save it as
`datasets/Battery_dataset.csv`.

## Reproducing the results

```bash
# Experiments E1–E6 (revision experiments)
python revision_experiments.py

# Main evaluation and sensitivity sweeps
python run_evaluation.py
python run_sensitivity.py

# Regenerate all figures from results/
python generate_figures.py
```

All stochastic experiments are run over 5 fixed seeds (42, 123, 456, 789,
1024); reported metrics are mean ± std over those seeds.

## License

MIT License — see [LICENSE](LICENSE).
