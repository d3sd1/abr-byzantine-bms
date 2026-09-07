# Applied Energy round 2 — experimental results (v3 signal model)

**Previous Ms.** APEN-D-26-09704R1 (REJECT with invitation to resubmit) · **Draft** `2026-abr-pinn-bms-byzantine`
**Run:** 2026-09-05 · **Seeds:** 42, 123, 456, 789, 1024 (5) · **Subsample:** 2
**Dataset:** El Tiemblo Solar+Storage LiFePO₄ 16s16p — 11 262 samples after subsampling, **808.3 h (33.7 days) of logged coverage** inside a 70-day calendar span (the log is event-driven and has gaps).
**Total runtime:** 2 225.7 s (37.1 min, CPU only) · every number is raw script output.

Results: `results/revision2_results.json` · consolidated under the `v3` key of `results/aggregated_results.json`.
Figures: `figures/figR6…figR9`, `figR1v3`, `figR2v3`, `figR5v3`.
Reproduce: `python revision2_experiments.py` then `python generate_figures_v3.py`.

> **Verdict in one paragraph.** The rebuilt validation supports the paper's *narrowed* claim and refutes part of the
> broad one. With signal-level fault injection and physically grounded module dispersion, the round-1 detector is
> shown to wrongly quarantine a healthy-but-imbalanced module and two kinds of healthy link **65 % of the time**,
> and to quarantine a healthy node in **8 %** of runs *with no fault present at all* — its reported zero false-positive
> rate was an artefact of the report-level injection. Adding the physical-consistency router drives all of those to
> **exactly 0** while keeping node-fault detection at 0.70. But three of the sixteen classes are simply not
> observable in report space (voltage-sensor offset, internal short circuit, thermal rise), one attack that round 1
> claimed as its best case (collusion) is **not detected at all**, and a trivial fixed threshold beats the detector on
> raw sensitivity at every dispersion level — it just pays for it with a 23–67 % false-exclusion rate.

---

## Two defects in the round-1 code, found while rebuilding

Both are fixed in the v3 modules; the round-1 scripts are left untouched as provenance.

| Defect | Evidence | Effect |
|---|---|---|
| **Current sign inverted.** `run_evaluation.py` integrates `dsoc = -eta*I*dt/(Q*3600)`, i.e. positive current = discharge. | Mean `I_pack` is **+146.4 A** over samples where the BMS SOC rises and **−65.8 A** where it falls; the windowed regression of ΔSOC on ∫I dt has positive slope at 50- and 200-sample windows. Positive current is **charging** (usual Victron convention). | Every Coulomb-counting baseline in round 1 integrated backwards. |
| **`dt` not recomputed after subsampling.** The round-1 code subsamples the frame but keeps the full-rate `dt`. | Horizon integrates to 630 h with the stale `dt` vs **808.3 h** with the recomputed one. | Charge throughput under-counted by ~1.3×; all drift terms scaled wrongly. |

---

## Two properties of the dataset that constrain what can be modelled (E10)

These are measurements, not modelling choices, and they are reported because they change what the paper may claim.

1. **The SOC channel is not consistent with the logged cell voltages.**
   `corr(inv_OCV(V_pack/16), SOC_bms) = −0.349` — negative. Where the BMS reports SOC ≈ **1.3 %**, the minimum
   and maximum cells sit at an OCV-implied **70.1 %** and **99.1 %**. Absolute voltage→SOC re-anchoring would drag
   every local estimate by 40–70 SOC points, and a voltage-triggered full-charge reset fires hundreds of times at
   low pack SOC. Neither is usable on this pack.
2. **The current channel is not consistent with the SOC channel either.**
   ∫I dt = **+1 554 Ah** net over the window while the SOC ends **9 points below** where it started (mean I_pack
   = +3.2 A). An open-loop Coulomb counter has ≈0.55 SOC RMSE against the reference for *any* assumed capacity
   (4480 / 9600 / 2000 Ah all give 0.55–0.60).

**Consequence for the model.** Each node runs a *differential* estimator: the common mode comes from the pack
shunt (which every node on the RS485 bus reads) and the node contributes its own module-level deviation,
integrated from (own current − pack current) over the nominal group capacity, plus a bounded differential OCV
correction that uses only the gap between its own cell voltage and the pack average cell voltage. This is what a
distributed BMS actually does, and it places the experiment where the paper's claim lives: detecting **node-level
deviations**, not absolute SOC.

---

## E7 — Fault-class discrimination matrix (R1.1, R2)

**Measured:** 16 fault classes (+ a fault-free control) × 8 detectors × 5 seeds. Faults are injected on a measured
signal (N1, N2, B1–B4) or on a message (N3–N8, G1–G3), never by editing the final SOC number except where that
*is* the fault (N5, N7). Figure `figR6_fault_class_matrix.pdf`; example traces in `figR9_aging_vs_byzantine.pdf`.

P(target node ends in each state), mean over 5 seeds:

| Class | expected | ABR **with** router (v3) | ABR **without** router (round 1) | ABR router-first | fixed 3 % (steps excl.) | Med/MAD (steps excl.) |
|---|---|---|---|---|---|---|
| N1 current offset +5 A | quarantine | **Q 0.80**, S 0.20 | Q 0.80 | Q 0.80 | 0.80 | 0.70 |
| N2 voltage offset +50 mV | quarantine | T 0.80, S 0.20 | T 0.80 | T 0.80 | 0.40 | 0.01 |
| N3 frozen report | quarantine | **Q 1.00** | Q 1.00 | Q 1.00 | 0.84 | 0.83 |
| N4 message corruption | quarantine | **Q 1.00** | Q 1.00 | Q 1.00 | 0.95 | 0.77 |
| N5 firmware bias +5 % | quarantine | **Q 0.80**, S 0.20 | Q 0.80 | Q 0.80 | 0.60 | 0.18 |
| N6 collusion (5 nodes, +3 %) | quarantine | T 0.96, S 0.04 | T 0.96 | **Q 1.00** | 0.13 | 0.00 |
| N7 stealth drift | quarantine | **Q 1.00** | Q 1.00 | Q 1.00 | 0.99 | 0.98 |
| N8 random report | quarantine | **Q 1.00** | Q 1.00 | Q 1.00 | 0.94 | 0.77 |
| G1 packet loss (bursty) | no quarantine | Q 0.00, **B 0.87** | **Q 0.80** | Q 0.00 | 0.23 | 0.01 |
| G2 message delay | no quarantine | Q 0.00, **B 1.00** | **Q 1.00** | Q 0.00 | 0.44 | 0.07 |
| G3 single-step spikes | no quarantine | Q 0.00, T 0.80 | Q 0.00 | **Q 1.00** | 0.34 | 0.00 |
| B1 aging (−15 % capacity) | no quarantine | Q 0.00, T 0.80 | Q 0.00 | Q 0.00 | 0.28 | 0.00 |
| B2 imbalance (−8 SOC pts) | no quarantine | Q 0.00, **B 0.80** | **Q 0.80** | Q 0.00 | 0.39 | 0.11 |
| B3 internal short 33 Ω | no quarantine | Q 0.00, **T 1.00** | Q 0.00 | Q 0.00 | 0.33 | 0.00 |
| B3b internal short, severe | no quarantine | Q 0.00, **T 1.00** | Q 0.00 | Q 0.00 | 0.37 | 0.00 |
| B4 thermal +5 °C | no quarantine | Q 0.00, **T 1.00** | Q 0.00 | Q 0.00 | 0.34 | 0.00 |
| **no fault (control)** | — | Q 0.00, T 1.00 | Q 0.00 | Q 0.00 | **0.34** | 0.00 |

Q = quarantined, B = battery-anomaly, S = suspect only, T = trusted.

False positives on the 15 healthy nodes, averaged over all 17 scenarios:

| Detector | mean FPR (steps) | P(a healthy node is ever excluded) | FPR in the **fault-free** control |
|---|---|---|---|
| **ABR with router (v3)** | **0.0000** | **0.000** | **0.0000 ± 0.0000** |
| ABR router-first | 0.0000 | 0.000 | 0.0000 ± 0.0000 |
| ABR without router (round 1) | 0.0120 | 0.097 | 0.0106 ± 0.0182 |
| Median/MAD z>3 | 0.0298 | 0.359 | 0.0306 ± 0.0198 |
| Fixed 5 % | 0.1402 | 0.474 | 0.1413 ± 0.0488 |
| Fixed 3 % | 0.2274 | 0.621 | 0.2241 ± 0.0429 |
| PBFT-BMS trimmed mean | 0.6244 | 0.997 | 0.6323 ± 0.0244 |

Detection latency (ABR with router, mean over seeds that detect): N8 **0.1 h**, N4 **1.1 h**, N7 **31.7 h**,
N3 **55.2 h**, N1 **143.1 h**, N5 **198.7 h**. Consensus RMSE against the true mean module SOC ranges
0.0081 (B1) to 0.0149 (N6).

**Conclusion (one line):** The router is what makes the central claim true — without it the detector quarantines an
imbalanced module, a lossy link and a delayed link in 80–100 % of runs, and a healthy node in 8 % of fault-free
runs; with it, **no battery fault and no glitch is ever quarantined (0.00 across 8 classes × 5 seeds) while 6 of the
8 node-fault classes are still caught**, but the two it misses (voltage offset, collusion) are missed completely.

---

## E8 — Honest-dispersion sweep (R1.2)

**Measured:** λ scales (σ_Q, σ_bI, σ_e0) jointly. Reported against the *measured* honest dispersion the model
produces, not against λ. Figure `figR7_dispersion_sweep.pdf`.

| λ | honest IQR of \|ŝ−med\| | p95 | cross-node MAD |
|---|---|---|---|
| 0.25 | 0.0111 ± 0.0011 | 0.0244 | 0.0091 |
| 0.5 | 0.0159 ± 0.0023 | 0.0429 | 0.0104 |
| **1.0** | **0.0232 ± 0.0078** | **0.0989** | **0.0126** |
| 2.0 | 0.0470 ± 0.0195 | 0.2165 | 0.0225 |
| 4.0 | 0.1209 ± 0.0327 | 0.3221 | 0.0432 |
| 8.0 | 0.2169 ± 0.0317 | 0.4521 | 0.0752 |

TPR = mean over {N5, N1}; "wrong" = mean over {B1, G1}, where any exclusion is an error; FPR on healthy nodes:

| λ | ABR TPR / wrong / FPR | fixed 3 % | fixed 5 % | Median/MAD |
|---|---|---|---|---|
| 0.25 | 0.530 / **0.000** / **0.000** | 0.837 / 0.019 / 0.028 | 0.536 / 0.008 / 0.007 | 0.638 / 0.011 / 0.005 |
| 0.5 | 0.529 / **0.000** / **0.000** | 0.740 / 0.092 / 0.108 | 0.563 / 0.009 / 0.035 | 0.536 / 0.009 / 0.012 |
| 1.0 | 0.542 / **0.000** / **0.000** | 0.702 / 0.257 / 0.226 | 0.563 / 0.164 / 0.140 | 0.442 / 0.008 / 0.028 |
| 2.0 | 0.265 / **0.000** / **0.000** | 0.825 / 0.414 / 0.367 | 0.701 / 0.276 / 0.259 | 0.168 / 0.017 / 0.029 |
| 4.0 | **0.000** / 0.000 / 0.000 | 0.653 / 0.549 / 0.541 | 0.560 / 0.458 / 0.420 | 0.050 / 0.012 / 0.034 |
| 8.0 | 0.085 / **0.000** / **0.000** | 0.822 / 0.647 / 0.670 | 0.745 / 0.582 / 0.595 | 0.020 / 0.034 / 0.059 |

**Conclusion (one line):** A fixed absolute threshold has **higher raw sensitivity than the adaptive detector at
every dispersion level**, so R1.2 is right that a large part of the round-1 result was "easy by physics"; what the
fixed threshold cannot do is stay quiet — its false-exclusion rate climbs from 2.8 % to 67 % as honest dispersion
grows from 1.1 % to 21.7 % SOC, while the ABR detector's stays at exactly 0 — **but it buys that by going silent,
detecting nothing at all once dispersion reaches 12 % IQR.**

---

## E9 — Fault magnitude vs honest spread (R1.2)

**Measured:** N5 bias swept 1–10 %, N1 current offset swept 2–10 A, λ = 1 (honest cross-node MAD = 0.0126 SOC).
Figure `figR8_magnitude_vs_spread.pdf`.

| N5 bias | in MAD units | ABR P(quarantine) | ABR steps excl. | latency | fixed 3 % steps excl. |
|---|---|---|---|---|---|
| 1 % | 0.79 | 0.00 | 0.000 | — | 0.295 |
| 2 % | 1.59 | 0.00 | 0.000 | — | 0.421 |
| 3 % | 2.38 | 0.00 | 0.000 | — | 0.574 |
| **5 %** | **3.97** | **0.80** | 0.530 | **198.7 h** | 0.604 |
| 10 % | 7.94 | 1.00 | 0.770 | 147.7 h | 0.843 |

| N1 offset | ABR P(quarantine) | steps excl. | latency |
|---|---|---|---|
| 2 A | 0.00 | 0.000 | — |
| 5 A | 0.80 | 0.555 | 143.1 h |
| 10 A | 1.00 | 0.858 | 56.0 h |

**Conclusion (one line):** The minimum detectable firmware bias lies between **2.4 and 4.0 cross-node MAD**
(3 % undetected, 5 % detected in 4/5 seeds), the minimum detectable current-sensor offset between **2 and 5 A**,
and detection of the marginal cases takes **6–8 days** of field time — the detector is a slow, conservative
integrity monitor, not a fast intrusion detector, and the paper should say so.

---

## E10 — Calibration against the measured envelope (R1.2)

**Measured:** the reconstructed module voltages are anchored inside the logged [V_min, V_max] envelope, so the
voltage spread matches the real pack by construction; the SOC dispersion is an *output* of the model and is
reported as such.

| Quantity | measured (dataset) | simulated λ=0.25 | simulated λ=1 | simulated λ=4 |
|---|---|---|---|---|
| cell-voltage spread, mean | 0.1781 V | 0.1784 V | 0.1790 V | 0.1795 V |
| cell-voltage spread, p95 | 0.6300 V | 0.6310 V | 0.6312 V | 0.6312 V |
| module SOC dispersion, IQR | — (not measurable) | 0.0111 | 0.0232 | 0.1209 |
| module SOC dispersion, p95 | — | 0.0244 | 0.0989 | 0.3221 |

Router thresholds calibrated on fault-free data, first 10 % of the horizon, 5 seeds, percentile 99.9, margin 1.5:
**th_cI = 4.791** (pooled honest p99.9 = 3.194; honest max in window = 3.695), **th_CUSUM = 3.8 × 10⁻⁵**
(honest max in window = 2.6 × 10⁻⁵), **slack = 9.74 × 10⁻⁴** (pooled honest p99.9 of the residual). The CUSUM
threshold generalises: the honest CUSUM over the **full** horizon reaches 4.4 × 10⁻⁵ in a separate check, i.e. the
same order as the calibration window, because the report quantisation error telescopes.

**Conclusion (one line):** The voltage envelope of the simulated pack reproduces the real one to within 0.1 % of a
volt at the mean and the p95, so the honest dispersion is anchored to a measured quantity — **but the SOC
dispersion cannot be validated against the dataset at all, because this pack only logs min/max cell voltages, and
those are not OCV-consistent with its own SOC channel.**

---

## E11 — Voltage-anchor sensitivity (replaces Table 11)

**Measured:** three voltage-reconstruction schemes — OCV-proportional (v3), rank-uniform, and the round-1
min/max interpolation ("legacy") — each with no fault, N5 and B1.

| Anchor | control RMSE | N5 RMSE | B1 RMSE | honest IQR | **N5 P(quarantine)** | B1 P(quarantine) |
|---|---|---|---|---|---|---|
| OCV-proportional (v3) | 0.00839 ± 0.00265 | 0.00970 ± 0.00337 | 0.00814 | 0.0232 | **0.80** | 0.00 |
| rank-uniform | 0.00830 ± 0.00281 | 0.01027 ± 0.00364 | 0.00802 | 0.0221 | **0.40** | 0.00 |
| legacy min/max (round 1) | 0.00822 ± 0.00227 | 0.01044 ± 0.00300 | 0.00794 | 0.0219 | **0.20** | 0.00 |

**Conclusion (one line):** The reviewer's Table 11 observation **survives on RMSE** — consensus accuracy still
varies by under 2 % relative across the three schemes — but it **fails on the decision**: whether the +5 % biased
node is caught at all drops from 0.80 to 0.20 as the per-module voltage detail is degraded, so the module-level
signal now demonstrably carries information the round-1 experiment could not see.

---

## E1' — Detector ablation on the v3 model

**Measured:** four variants across four node-fault classes plus four classes that must **not** be quarantined.
Figure `figR1v3_detector_ablation.pdf`.

Detection of true node faults (fraction of post-onset steps the faulty node is excluded):

| Variant | N5 bias | N8 random | N7 drift | N6 collusion | FPR |
|---|---|---|---|---|---|
| EWMA trust only | 0.000 | 0.000 | 0.000 | 0.000 | 0.0000 |
| Gap only (no trust grading) | 0.530 | 1.000 | 0.984 | 0.000 | 0.0110 |
| EWMA + gap (round-1 detector) | 0.530 | 1.000 | 0.984 | 0.000 | 0.0110 |
| **EWMA + gap + router (v3)** | 0.530 | 1.000 | 0.908 | 0.000 | **0.0000** |

Classes where **any** quarantine is an error — P(target wrongly quarantined):

| Variant | B1 aging | B2 imbalance | G1 packet loss | G2 delay | mean |
|---|---|---|---|---|---|
| EWMA trust only | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| Gap only | 0.00 | 0.80 | 0.80 | 1.00 | 0.65 |
| **EWMA + gap (round-1 detector)** | 0.00 | **0.80** | **0.80** | **1.00** | **0.65** |
| **EWMA + gap + router (v3)** | 0.00 | **0.00** | **0.00** | **0.00** | **0.00** |

**Conclusion (one line):** The gap-quarantine is still what does the detecting (EWMA-only detects nothing), the
trust grading contributes nothing measurable on this model (gap-only and EWMA+gap are identical to three
decimals), and **the router is the component that earns the paper's central claim: it takes the wrong-quarantine
rate from 0.65 to 0.00 at a cost of 0.076 TPR on one class (N7).**

---

## E2' — Gap-threshold sweep on the v3 model

**Measured:** δ ∈ [1, 7] over seven classes (four node faults, two battery faults, one glitch). Figure
`figR2v3_gap_threshold_sweep.pdf`.

| δ | TPR (node faults) | FPR (healthy) | P(quar. \| node fault) | P(quar. \| **not** a node fault) | RMSE |
|---|---|---|---|---|---|
| 1 | 0.610 | 0.0000 | 0.70 | **0.00** | 0.0102 |
| 2 | 0.610 | 0.0000 | 0.70 | **0.00** | 0.0103 |
| 3 (default) | 0.610 | 0.0000 | 0.70 | **0.00** | 0.0104 |
| 4 | 0.521 | 0.0000 | 0.60 | **0.00** | 0.0105 |
| 5 | 0.477 | 0.0000 | 0.50 | **0.00** | 0.0104 |
| 6 | 0.477 | 0.0000 | 0.50 | **0.00** | 0.0104 |
| 7 | 0.477 | 0.0000 | 0.50 | **0.00** | 0.0104 |

**Conclusion (one line):** With the router in place the false-quarantine rate is **structurally zero across the whole
δ range**, so the zero-FPR result is not a cautious tuning choice; detection degrades monotonically above δ = 3,
which is therefore the sensible operating point and not a fitted one.

---

## E6' — Centralised vs distributed on the v3 model

| Attack | centralised mean fusion | distributed ABR (v3) | degradation |
|---|---|---|---|
| N5 firmware bias | 0.0107 | 0.0097 | **1.1×** |
| N8 random report | 0.0229 | 0.0093 | **2.5×** |
| N7 stealth drift | 0.0403 | 0.0096 | **4.2×** |
| N6 collusion | 0.0149 | 0.0149 | **1.0×** |

**Conclusion (one line):** The advantage of the distributed scheme is real but **far smaller than round 1 reported
(1.0×–4.2× here vs 8×–98× there)**, because the round-1 attacks were enormous by construction; under collusion,
which the detector does not catch, the distributed scheme gives **no benefit whatsoever**.

---

## Unfavourable results

Listed explicitly, because several of them narrow the paper's claims.

1. **Internal short circuit is invisible in report space (answers R2 negatively).** The specified 33 Ω short draws
   0.1 A, which at the declared 4480 Ah group capacity is 0.054 %/day. A severity-matched variant (B3b, 2.06 Ω,
   1.6 A ≈ 0.86 %/day) drives the module's **true** SOC 23 points away from its peers by the end of the horizon,
   yet the node's report barely moves (≤ 3.9 % shift) because the leakage current never crosses the node's shunt.
   Both B3 and B3b end **trusted in 5/5 seeds**. The detector does not raise a false alarm — and it does not
   diagnose the fault either. The manuscript must not claim short-circuit detection.
2. **Thermal fault is completely invisible.** B4 shifts the report by exactly 0.0000, because temperature does not
   enter the SOC estimator. Trusted in 5/5 seeds.
3. **Voltage-sensor offset (N2) is undetectable.** +50 mV on the LiFePO₄ plateau moves the report by at most
   0.7 % SOC; ABR leaves it trusted in 4/5 seeds. This is a *node* fault the method is supposed to catch and does not.
4. **Collusion (N6) is not detected.** Five nodes broadcasting true SOC +3 % are trusted in 4.8/5 seeds by both the
   v3 detector and the round-1 detector: their deviation (2.4 % mean) inflates the MAD and never crosses the gap.
   Round 1 reported collusion as its *best* case (98× advantage over centralised fusion) — that was entirely an
   artefact of injecting a flat 0.80 SOC. Only the `router-first` variant catches it, and that variant then wrongly
   quarantines the single-step-spike glitch (G3) in 5/5 seeds. There is no configuration tested that gets both.
5. **Capacity fade (B1) is invisible too, so the headline R1.1 result holds for a weaker reason than claimed.**
   A Coulomb-counting node reports its own current faithfully regardless of its capacity, so 15 % fade shifts the
   report by ≤ 0.7 % even though the module's true SOC drifts 5.7 points. Aging is not confused with an attack
   mostly because it does not look like anything. **B2 (imbalance) is the only battery fault that is both visible and
   correctly routed** (battery-anomaly in 4/5 seeds, quarantined in 0/5).
6. **A fixed absolute threshold has higher raw sensitivity than the detector at every dispersion level** (E8).
   R1.2's "easy by physics" objection is partly upheld: the adaptive machinery does not buy sensitivity, it buys
   silence on healthy nodes.
7. **The detector goes silent at high honest dispersion.** At λ = 4 (IQR 12.1 %) its TPR is 0.000. It keeps FPR at
   zero by not firing at all. The operating envelope is roughly IQR ≤ 5 % SOC.
8. **Detection is slow.** 143 h (N1 at 5 A) and 199 h (N5 at 5 %) — six to eight days of field time.
9. **Consensus RMSE is an order of magnitude worse than round 1** (0.0081–0.0149 here vs 0.0017 there), because
   in round 1 the reports *were* the ground truth plus N(0, 0.005) noise. The round-1 accuracy figures should not
   be carried over.
10. **The round-1 detector's reported zero false-positive rate does not survive.** On the v3 model it is 0.0120
    overall and **0.0106 in the fault-free control**, quarantining a healthy node in 8 % of runs with no fault present.
11. **Glitches are routed to the wrong label.** G1 and G2 are correctly *not* quarantined but are flagged
    `battery-anomaly` (0.87 and 1.00), which is a misclassification: the router only has two outcomes and a
    degraded link matches neither. A comms-health outcome is missing from the design.
12. **PBFT-BMS trimmed mean has a 62 % false-exclusion rate by construction** (it trims 10 of 16 nodes every
    step). It is included for completeness, but the comparison is structural rather than informative.
13. **Table 11's invariance is only half-answered.** Consensus RMSE still varies by under 2 % relative across the
    three anchoring schemes (E11); only the *detection decision* changes.
14. **Trust grading contributes nothing measurable.** Gap-only and EWMA+gap give identical TPR/FPR to three
    decimals on all four ablation classes. The round-1 claim that the asymmetric EWMA "keeps FPR at exactly 0"
    does not reproduce — the router does that.

---

## Mapping reviewer points → experiment → key number

| Reviewer point | Answered by | Key number |
|---|---|---|
| **R1.1** faults injected at report level; aging-vs-attack never tested | **E7**, **E1'**, figR9 | Signal-level injection across 16 classes. Round-1 detector wrongly quarantines B2/G1/G2 at **0.80/0.80/1.00**; with the router, **0.00/0.00/0.00**, while 6 of 8 node-fault classes are still caught. Aged module (B1): report shift ≤ 0.7 % SOC vs 5.7 points of true drift → trusted 4/5 |
| **R1.1** glitches must not be quarantined | **E7** | G1, G2, G3 quarantined **0.00** by the v3 detector (vs 0.80, 1.00, 0.00 without the router) |
| **R1.2** modules are near-copies; detection is easy by physics | **E8**, **E9**, **E10** | Honest dispersion is now an output: IQR **2.3 %**, p95 **9.9 %**, MAD **1.3 %** at λ=1, against a voltage envelope matching the measured one (0.179 V vs 0.178 V mean). Minimum detectable bias **2.4–4.0 MAD**; a fixed 3 % threshold reaches higher TPR but excludes a healthy node in **22–67 %** of steps (**34 %** with no fault at all) |
| **R1.2** compare against a trivial fixed threshold | **E7**, **E8** | Fixed 3 % / fixed 5 % / Median-MAD / PBFT included at every λ. Fixed 3 % wins on TPR, loses on FPR by 22–67 points |
| **R2** no evidence on **internal short circuit** | **E7** (B3, B3b) | Trusted **5/5 seeds** at both severities; 1.6 A leakage moves true SOC 23 points but the report ≤ 3.9 % — **the method cannot diagnose it, and the paper must not claim it** |
| **R2** no evidence on **cell imbalance** | **E7** (B2), **E1'** | −8 SOC-point imbalance: **battery-anomaly 0.80**, quarantined **0.00** with the router; quarantined **0.80** without it |
| **R2** claims are broader than the evidence | **Unfavourable results** §1–14 | Collusion undetected (0.00), voltage offset undetected (0.00), thermal invisible, centralised-vs-distributed advantage 1.0×–4.2× not 8×–98× |
| **Table 11** per-module detail does not contribute | **E11** | RMSE still nearly invariant (0.00822–0.00839, <2 % relative), but N5 detection falls **0.80 → 0.40 → 0.20** across OCV / rank / legacy anchoring |
| Detector component contributions | **E1'**, **E2'** | Gap does the detecting; trust grading adds nothing measurable; router takes wrong-quarantine from **0.65 → 0.00**; FPR is 0 across δ ∈ [1,7] |

**Net.** Every reviewer point that can be settled by computation is settled with real 5-seed results on 808 h of
field data. Two of the settlements are negative — the method does not detect internal short circuit, thermal
faults, voltage-sensor offset or low-amplitude collusion — and the manuscript's claims must be narrowed to the
one thing the evidence does support: **separating node and communication faults from physically consistent
battery divergence, without false quarantines, at the cost of slow detection and a bounded operating envelope.**
