#!/usr/bin/env python3
"""
Figure regeneration entry-point for the ABR-PINN paper (Applied Energy revision).

Regenerates ALL publication figures from the saved result JSONs without
re-running the (long) simulations:

  Main evaluation figures   (from results/abr_pinn_results.json + sweeps)
      run_evaluation.py / run_sensitivity.py own these; re-run those scripts
      to regenerate fig1/fig2/fig3/fig_violin/fig_convergence.

  Revision figures          (from results/revision_results.json)
      figR1_detector_ablation.pdf          E1 (R1.2)
      figR2_fpr_threshold_sweep.pdf        E2 (R1.3)
      figR3_per_module_soc.pdf             E3 (R1.4/R2.6/R2.8)
      figR4_pinn_vs_filter.pdf             E4 (R2.5/R2.10)
      figR5_centralised_vs_distributed.pdf E6 (R2.9)

All figures: PDF, serif (Latin Modern / CM), 10 pt, colour-blind-safe palette.
Run:  python generate_figures.py
"""

import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

RESULTS = SCRIPT_DIR / "results" / "revision_results.json"


def main():
    import revision_experiments as rev

    if not RESULTS.exists():
        raise SystemExit(
            f"{RESULTS} not found -- run revision_experiments.py first.")

    with open(RESULTS) as f:
        data = json.load(f)

    print("Regenerating revision figures from", RESULTS.name)
    rev.make_figures(data)
    print("Done. Figures written to", (SCRIPT_DIR / 'figures'))


if __name__ == '__main__':
    main()
