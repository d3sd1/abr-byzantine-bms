#!/usr/bin/env python3
"""
Sensitivity & Scalability Sweeps + Figure Generation for ABR-PINN paper.

Three experiments:
1. lambda_phys sweep: vary PINN physics-loss weight {0.01, 0.1, 1.0, 10.0}
   under constant_bias attack. Trains actual PINN with each weight and
   blends its SOC prediction with Coulomb counting.

2. w1 sweep: vary corroboration-score physics weight {0.3, 0.5, 0.7, 0.9}
   under slow_drift attack. Corroboration score modulates EWMA
   accumulation rate (faithful to Alg. 1 Phase 2).

3. Scalability: vary f in {1, 2, 3, 5} across all attacks for ABR-PINN,
   ABR-Stat, PBFT-BMS.

All sweeps use 10 seeds (sufficient for sensitivity analysis).

Author: Andrei Garcia Cuadra (ETSIDI-UPM)
"""

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams

warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_evaluation import (
    load_battery_data, simulate_modules, inject_attacks,
    compute_metrics, PhysicsModel, CoulombCounting, SimplePINN,
    ABR_Stat, PBFT_BMS,
    ocv_cell, DATA_FILE, OUTPUT_DIR,
    N_MODULES, F_BYZANTINE, EPSILON_BAR, GAMMA_HIGH, GAMMA_LOW,
    W1, W2, N_SERIES, Q_NOM_CELL, R0_CELL
)

rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Latin Modern Roman', 'Computer Modern Roman', 'Times New Roman']
rcParams['font.size'] = 10
rcParams['axes.labelsize'] = 11
rcParams['legend.fontsize'] = 9
rcParams['figure.dpi'] = 150

SWEEP_SEEDS = 10
SUBSAMPLE = 4


class ABR_PINN_Corr:
    """ABR-PINN with corroboration-score-modulated trust dynamics.

    Faithful to Algorithm 1 in the paper:
    Phase 2 computes corroboration score gamma_ij = w1*g(rho) + w2*h(Delta).
    gamma modulates EWMA accumulation rate:
      - Low gamma (physics-implausible) -> faster EWMA rise -> earlier quarantine
      - High gamma (physics-plausible) -> slower EWMA rise -> protects honest nodes
    """

    EWMA_ALPHA_BASE = 0.10
    EWMA_ALPHA_DOWN = 0.30
    TRUST_SIGMOID_K = 1.5
    TRUST_SIGMOID_X0 = 6.0
    EWMA_CAP = 25.0
    MAD_FLOOR = 0.005
    GAP_THRESHOLD = 3.0
    MIN_EWMA_QUARANTINE = 5.0

    def __init__(self, n_modules, f_max, epsilon_bar, gamma_high, gamma_low,
                 w1, w2, physics_models):
        self.n = n_modules
        self.f = f_max
        self.eps = epsilon_bar
        self.gh = gamma_high
        self.gl = gamma_low
        self.w1 = w1
        self.w2 = w2
        self.models = physics_models
        self.trust = np.ones(n_modules) * gamma_high
        self.quarantined = np.zeros(n_modules, dtype=bool)
        self.ewma_z = np.zeros(n_modules)
        self.prev_reports = None
        self.history_trust = []

    def reset(self):
        self.trust = np.ones(self.n) * self.gh
        self.quarantined = np.zeros(self.n, dtype=bool)
        self.ewma_z = np.zeros(self.n)
        self.prev_reports = None
        self.history_trust = []

    def _corroboration_score(self, j, soc_j, v_j, i_j, t_j, dt):
        """Compute gamma_ij = w1 * g(rho) + w2 * h(Delta) for node j."""
        # g(rho): physics plausibility (OCV cross-validation)
        v_pred = self.models[j].predict_voltage(soc_j, i_j, t_j)
        rho = abs(v_j - v_pred)
        tau_rho = 0.03  # temperature for voltage residual (V)
        g_score = np.exp(-rho / tau_rho)

        # h(Delta): temporal rate-of-change plausibility
        if self.prev_reports is not None:
            delta = abs(soc_j - self.prev_reports[j]) / max(dt, 1e-6)
            delta_max = 1.0 / 3600  # 1C max rate
            tau_delta = delta_max * 0.5
            h_score = 1.0 / (1.0 + np.exp((delta - delta_max) / tau_delta))
        else:
            h_score = 1.0

        gamma = self.w1 * g_score + self.w2 * h_score
        return np.clip(gamma, 0.0, 1.0)

    def step(self, soc_reports, v_obs, i_obs, t_obs, dt=1.0):
        n = self.n
        detected = []

        # --- Phase 1: Statistical z-scores ---
        active = ~self.quarantined
        n_active = active.sum()
        if n_active > 0:
            active_reports = soc_reports[active]
            median_soc = np.median(active_reports)
            deviations = np.abs(soc_reports - median_soc)
            mad = max(np.median(np.abs(active_reports - median_soc)),
                      self.MAD_FLOOR)
            z_scores = deviations / (1.4826 * mad)
        else:
            z_scores = np.zeros(n)

        # --- Phase 2: Corroboration-modulated EWMA ---
        for j in range(n):
            if not self.quarantined[j]:
                gamma_j = self._corroboration_score(
                    j, soc_reports[j], v_obs[j], i_obs[j], t_obs[j], dt)

                # Modulate EWMA alpha by corroboration score:
                # Low gamma -> faster accumulation (2x at gamma=0)
                # High gamma -> slower accumulation (0.5x at gamma=1)
                corr_factor = 2.0 - 1.5 * gamma_j  # [0.5, 2.0]

                if z_scores[j] >= self.ewma_z[j]:
                    alpha_eff = self.EWMA_ALPHA_BASE * corr_factor
                else:
                    # Recovery: high gamma -> faster recovery
                    alpha_eff = self.EWMA_ALPHA_DOWN * (0.5 + 0.5 * gamma_j)

                alpha_eff = np.clip(alpha_eff, 0.01, 0.5)
                self.ewma_z[j] = min(
                    (1 - alpha_eff) * self.ewma_z[j] + alpha_eff * z_scores[j],
                    self.EWMA_CAP)
                self.trust[j] = self.gl + (self.gh - self.gl) / (
                    1.0 + np.exp(self.TRUST_SIGMOID_K *
                                 (self.ewma_z[j] - self.TRUST_SIGMOID_X0)))

        # --- Phase 3: Gap-based quarantine ---
        n_quarantined = int(self.quarantined.sum())
        if n_quarantined < self.f:
            active_idx = np.where(~self.quarantined)[0]
            if len(active_idx) > self.f + 1:
                ewma_active = self.ewma_z[active_idx]
                order = np.argsort(-ewma_active)
                sorted_ewma = ewma_active[order]
                max_gap = 0.0
                gap_pos = -1
                limit = min(self.f, len(sorted_ewma) - 1)
                for i in range(limit):
                    gap = sorted_ewma[i] - sorted_ewma[i + 1]
                    if gap > max_gap:
                        max_gap = gap
                        gap_pos = i
                if (max_gap > self.GAP_THRESHOLD and
                        sorted_ewma[0] > self.MIN_EWMA_QUARANTINE):
                    for i in range(gap_pos + 1):
                        node = active_idx[order[i]]
                        if not self.quarantined[node] and n_quarantined < self.f:
                            detected.append(node)
                            self.quarantined[node] = True
                            n_quarantined += 1

        # --- Phase 4: Trust-weighted consensus ---
        weights = np.zeros(n)
        for i in range(n):
            if not self.quarantined[i]:
                weights[i] = self.trust[i]
        w_sum = weights.sum()
        if w_sum < 1e-10:
            active_mask = ~self.quarantined
            if active_mask.any():
                weights[active_mask] = 1.0
                w_sum = weights.sum()
            else:
                self.prev_reports = soc_reports.copy()
                return np.median(soc_reports), detected
        weights /= w_sum
        consensus_soc = np.sum(weights * soc_reports)

        self.prev_reports = soc_reports.copy()
        self.history_trust.append(self.trust.copy())
        return consensus_soc, detected


def run_abr_trial(df, attack_type, byzantine_nodes, seed, w1=0.7,
                  lambda_phys=1.0, f_byz=F_BYZANTINE, subsample=SUBSAMPLE,
                  use_pinn_blend=False):
    """Run ABR-PINN trial with configurable parameters."""
    rng = np.random.default_rng(seed)

    if subsample > 1:
        df_sub = df.iloc[::subsample].reset_index(drop=True)
    else:
        df_sub = df

    T_data = len(df_sub)
    soc_gt, V_pack, I_pack, T_batt, dt, module_data = simulate_modules(
        df_sub, N_MODULES, rng)

    t_start = T_data // 4
    soc_reports_all = np.column_stack([m['SOC'] for m in module_data])

    # If PINN blend is enabled, train PINN and blend into SOC reports
    if use_pinn_blend:
        V_cell_avg = np.mean([m['V_cell'] for m in module_data], axis=0)
        I_avg = np.mean([m['I'] for m in module_data], axis=0)
        T_avg = np.mean([m['T'] for m in module_data], axis=0)
        pinn = SimplePINN(hidden=32, rng=np.random.default_rng(seed))
        n_train = min(2000, T_data // 2)
        pinn.train(V_cell_avg[:n_train], I_avg[:n_train],
                   T_avg[:n_train], soc_gt[:n_train],
                   dt[:n_train], max_iter=150, lambda_phys=lambda_phys)
        soc_pinn = pinn.predict_soc(V_cell_avg, I_avg, T_avg)
        alpha_blend = 0.3
        for m_idx in range(N_MODULES):
            noise = rng.normal(0, 0.003, T_data)
            soc_reports_all[:, m_idx] = (
                (1 - alpha_blend) * soc_reports_all[:, m_idx]
                + alpha_blend * (soc_pinn + noise))
            soc_reports_all[:, m_idx] = np.clip(soc_reports_all[:, m_idx], 0, 1)

    soc_attacked, attack_mask = inject_attacks(
        soc_reports_all, attack_type, byzantine_nodes, t_start, rng)

    physics_models = [PhysicsModel(r0=m['R0']) for m in module_data]
    v_obs_all = np.column_stack([m['V_cell'] for m in module_data])
    i_obs_all = np.column_stack([m['I'] for m in module_data])
    t_obs_all = np.column_stack([m['T'] for m in module_data])

    w2 = 1.0 - w1
    abr = ABR_PINN_Corr(N_MODULES, f_byz, EPSILON_BAR,
                         GAMMA_HIGH, GAMMA_LOW, w1, w2, physics_models)
    abr.reset()

    soc_abrpinn = np.zeros(T_data)
    det_abrpinn = []
    for t in range(T_data):
        soc_abrpinn[t], det = abr.step(
            soc_attacked[t], v_obs_all[t], i_obs_all[t],
            t_obs_all[t], dt=dt[t])
        det_abrpinn.append(list(np.where(abr.quarantined)[0]))

    metrics = compute_metrics(
        soc_abrpinn, soc_gt, det_abrpinn, byzantine_nodes,
        attack_mask, t_start)

    return metrics, abr


def run_baseline_trial(df, attack_type, byzantine_nodes, seed, method,
                       f_byz=F_BYZANTINE, subsample=SUBSAMPLE):
    rng = np.random.default_rng(seed)
    if subsample > 1:
        df_sub = df.iloc[::subsample].reset_index(drop=True)
    else:
        df_sub = df

    T_data = len(df_sub)
    soc_gt, V_pack, I_pack, T_batt, dt, module_data = simulate_modules(
        df_sub, N_MODULES, rng)
    t_start = T_data // 4
    soc_reports_all = np.column_stack([m['SOC'] for m in module_data])
    soc_attacked, attack_mask = inject_attacks(
        soc_reports_all, attack_type, byzantine_nodes, t_start, rng)

    if method == 'ABR-Stat':
        m = ABR_Stat(N_MODULES, f_byz)
        soc_out = np.zeros(T_data)
        det_out = []
        for t in range(T_data):
            soc_out[t], det = m.step(soc_attacked[t])
            det_out.append(det)
    elif method == 'PBFT-BMS':
        m = PBFT_BMS(N_MODULES, f_byz)
        soc_out = np.zeros(T_data)
        det_out = []
        for t in range(T_data):
            soc_out[t], det = m.step(soc_attacked[t])
            det_out.append(det)
    else:
        raise ValueError(f"Unknown method: {method}")

    return compute_metrics(soc_out, soc_gt, det_out, byzantine_nodes,
                           attack_mask, t_start)


# ===================================================================
# Sweep 1: lambda_phys (PINN training weight)
# ===================================================================

def sweep_lambda_phys(df):
    """Sweep lambda_phys under constant_bias attack.

    lambda_phys controls PINN physics-loss weight during training.
    Higher values produce more physically consistent SOC predictions.
    We train a PINN per lambda and blend with Coulomb counting (alpha=0.3).
    """
    print("\n" + "=" * 60)
    print("SWEEP 1: lambda_phys in {0.01, 0.1, 1.0, 10.0}")
    print("  Attack: constant_bias | w1=0.7 (fixed)")
    print("=" * 60)

    lambdas = [0.01, 0.1, 1.0, 10.0]
    attack = 'constant_bias'
    byzantine_nodes = [0, 3, 7, 11, 14]
    results = {}

    for lam in lambdas:
        print(f"\n  lambda_phys = {lam}")
        seed_metrics = []
        for seed in range(SWEEP_SEEDS):
            t0 = time.time()
            m, _ = run_abr_trial(df, attack, byzantine_nodes, seed,
                                 w1=W1, lambda_phys=lam,
                                 use_pinn_blend=True)
            seed_metrics.append(m)
            if seed % 5 == 0:
                print(f"    Seed {seed:2d}: RMSE={m['rmse_attack']:.4f} "
                      f"FPR={m['fpr']:.4f} TPR={m['tpr']:.3f} "
                      f"({time.time()-t0:.1f}s)")

        results[lam] = {
            'fpr_mean': float(np.mean([m['fpr'] for m in seed_metrics])),
            'fpr_std': float(np.std([m['fpr'] for m in seed_metrics])),
            'tpr_mean': float(np.mean([m['tpr'] for m in seed_metrics])),
            'tpr_std': float(np.std([m['tpr'] for m in seed_metrics])),
            'rmse_mean': float(np.mean([m['rmse_attack'] for m in seed_metrics])),
            'rmse_std': float(np.std([m['rmse_attack'] for m in seed_metrics])),
        }
        r = results[lam]
        print(f"    => FPR={r['fpr_mean']:.4f} TPR={r['tpr_mean']:.3f} "
              f"RMSE={r['rmse_mean']:.4f}+/-{r['rmse_std']:.4f}")

    return results


# ===================================================================
# Sweep 2: w1 (corroboration score weight)
# ===================================================================

def sweep_w1(df):
    """Sweep w1 under slow_drift attack.

    w1 weights the physics plausibility g(rho) in the corroboration
    score gamma = w1*g(rho) + w2*h(Delta). The corroboration score
    modulates EWMA accumulation rate: low gamma -> faster rise.
    """
    print("\n" + "=" * 60)
    print("SWEEP 2: w1 in {0.3, 0.5, 0.7, 0.9}")
    print("  Attack: slow_drift | lambda_phys=1.0 (fixed)")
    print("=" * 60)

    w1_values = [0.3, 0.5, 0.7, 0.9]
    attack = 'slow_drift'
    byzantine_nodes = [0, 3, 7, 11, 14]
    results = {}

    for w1_val in w1_values:
        print(f"\n  w1 = {w1_val}")
        seed_metrics = []
        for seed in range(SWEEP_SEEDS):
            t0 = time.time()
            m, _ = run_abr_trial(df, attack, byzantine_nodes, seed,
                                 w1=w1_val)
            seed_metrics.append(m)
            if seed % 5 == 0:
                lat = m['detection_latency']
                lat_str = f"{lat}" if lat is not None else "N/A"
                print(f"    Seed {seed:2d}: FPR={m['fpr']:.4f} TPR={m['tpr']:.3f} "
                      f"Lat={lat_str} ({time.time()-t0:.1f}s)")

        det_lats = [m['detection_latency'] for m in seed_metrics
                    if m['detection_latency'] is not None]
        results[w1_val] = {
            'fpr_mean': float(np.mean([m['fpr'] for m in seed_metrics])),
            'fpr_std': float(np.std([m['fpr'] for m in seed_metrics])),
            'tpr_mean': float(np.mean([m['tpr'] for m in seed_metrics])),
            'tpr_std': float(np.std([m['tpr'] for m in seed_metrics])),
            'latency_mean': float(np.mean(det_lats)) if det_lats else None,
            'latency_std': float(np.std(det_lats)) if det_lats else None,
            'rmse_mean': float(np.mean([m['rmse_attack'] for m in seed_metrics])),
            'rmse_std': float(np.std([m['rmse_attack'] for m in seed_metrics])),
        }
        r = results[w1_val]
        lat_str = f"{r['latency_mean']:.0f}" if r['latency_mean'] else "N/A"
        print(f"    => FPR={r['fpr_mean']:.4f} TPR={r['tpr_mean']:.3f} Lat={lat_str}")

    return results


# ===================================================================
# Sweep 3: Scalability
# ===================================================================

def sweep_scalability(df):
    print("\n" + "=" * 60)
    print("SWEEP 3: Scalability f in {1, 2, 3, 5}")
    print("=" * 60)

    f_values = [1, 2, 3, 5]
    attack_types = ['constant_bias', 'random_noise', 'slow_drift', 'collusion']
    all_byzantine = [0, 3, 7, 11, 14]
    methods_list = ['ABR-PINN', 'ABR-Stat', 'PBFT-BMS']
    results = {}

    for f_val in f_values:
        byz_nodes = all_byzantine[:f_val]
        results[f_val] = {}
        print(f"\n  f = {f_val}, Byzantine nodes: {byz_nodes}")

        for atk in attack_types:
            results[f_val][atk] = {}
            for method in methods_list:
                seed_metrics = []
                for seed in range(SWEEP_SEEDS):
                    if method == 'ABR-PINN':
                        m, _ = run_abr_trial(df, atk, byz_nodes, seed,
                                             f_byz=f_val)
                    else:
                        m = run_baseline_trial(df, atk, byz_nodes, seed,
                                               method, f_byz=f_val)
                    seed_metrics.append(m)

                results[f_val][atk][method] = {
                    'fpr_mean': float(np.mean([m['fpr'] for m in seed_metrics])),
                    'tpr_mean': float(np.mean([m['tpr'] for m in seed_metrics])),
                    'rmse_mean': float(np.mean([m['rmse_attack'] for m in seed_metrics])),
                }

            abr = results[f_val][atk]['ABR-PINN']
            print(f"    {atk}: ABR-PINN FPR={abr['fpr_mean']:.4f} "
                  f"TPR={abr['tpr_mean']:.3f} RMSE={abr['rmse_mean']:.4f}")

    return results


# ===================================================================
# Figure generation
# ===================================================================

def generate_violin_figure(df):
    """Trust score distributions under different attacks."""
    print("\n  Generating violin plot (trust distributions)...")

    from run_evaluation import ABR_PINN as ABR_PINN_Orig

    attack_types = ['constant_bias', 'random_noise', 'slow_drift', 'collusion']
    attack_labels = ['Constant\nBias', 'Random\nNoise', 'Slow\nDrift', 'Collusion']
    byzantine_nodes = [0, 3, 7, 11, 14]
    byz_set = set(byzantine_nodes)

    fig, axes = plt.subplots(1, 4, figsize=(12, 4), sharey=True)

    for idx, (atk, label) in enumerate(zip(attack_types, attack_labels)):
        rng = np.random.default_rng(42)
        df_sub = df.iloc[::SUBSAMPLE].reset_index(drop=True)
        T_data = len(df_sub)
        soc_gt, V_pack, I_pack, T_batt, dt_arr, module_data = simulate_modules(
            df_sub, N_MODULES, rng)
        t_start = T_data // 4
        soc_reports_all = np.column_stack([m['SOC'] for m in module_data])
        soc_attacked, attack_mask = inject_attacks(
            soc_reports_all, atk, byzantine_nodes, t_start, rng)
        physics_models = [PhysicsModel(r0=m['R0']) for m in module_data]
        v_obs_all = np.column_stack([m['V_cell'] for m in module_data])
        i_obs_all = np.column_stack([m['I'] for m in module_data])
        t_obs_all = np.column_stack([m['T'] for m in module_data])

        abr = ABR_PINN_Orig(N_MODULES, F_BYZANTINE, EPSILON_BAR,
                             GAMMA_HIGH, GAMMA_LOW, W1, W2, physics_models)
        abr.reset()
        for t in range(T_data):
            abr.step(soc_attacked[t], v_obs_all[t], i_obs_all[t],
                     t_obs_all[t], dt=dt_arr[t])

        trust_history = np.array(abr.history_trust)
        attack_start = t_start
        if len(trust_history) > attack_start + 100:
            trust_attack = trust_history[attack_start+100:]
            honest_trust = []
            byz_trust = []
            for j in range(N_MODULES):
                if j in byz_set:
                    byz_trust.extend(trust_attack[:, j].tolist())
                else:
                    honest_trust.extend(trust_attack[:, j].tolist())

            ax = axes[idx]
            parts = ax.violinplot([honest_trust, byz_trust],
                                  positions=[1, 2], showmeans=True,
                                  showmedians=True)
            for i, pc in enumerate(parts['bodies']):
                pc.set_facecolor(['#377eb8', '#e41a1c'][i])
                pc.set_alpha(0.7)
            parts['cmeans'].set_color('black')
            parts['cmedians'].set_color('gray')
            ax.set_xticks([1, 2])
            ax.set_xticklabels(['Honest', 'Byzantine'], fontsize=9)
            ax.set_title(label, fontsize=11)
            if idx == 0:
                ax.set_ylabel('Trust score $\\gamma_j$', fontsize=11)
            ax.set_ylim(-0.05, 1.05)
            ax.axhline(y=GAMMA_LOW, color='gray', linestyle=':', linewidth=0.8)
            ax.grid(True, alpha=0.2, axis='y')

    fig.suptitle('Trust Score Distributions During Attack Period', fontsize=13, y=1.02)
    fig.tight_layout()
    path = OUTPUT_DIR / "fig_violin_trust.pdf"
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"    Saved: {path}")


def generate_convergence_figure(df):
    """EWMA z-score convergence for Byzantine vs honest nodes."""
    print("\n  Generating convergence plot...")

    from run_evaluation import ABR_PINN as ABR_PINN_Orig

    rng = np.random.default_rng(42)
    byzantine_nodes = [0, 3, 7, 11, 14]

    df_sub = df.iloc[::SUBSAMPLE].reset_index(drop=True)
    T_data = len(df_sub)
    soc_gt, V_pack, I_pack, T_batt, dt_arr, module_data = simulate_modules(
        df_sub, N_MODULES, rng)
    t_start = T_data // 4
    soc_reports_all = np.column_stack([m['SOC'] for m in module_data])
    soc_attacked, attack_mask = inject_attacks(
        soc_reports_all, 'constant_bias', byzantine_nodes, t_start, rng)
    physics_models = [PhysicsModel(r0=m['R0']) for m in module_data]
    v_obs_all = np.column_stack([m['V_cell'] for m in module_data])
    i_obs_all = np.column_stack([m['I'] for m in module_data])
    t_obs_all = np.column_stack([m['T'] for m in module_data])

    abr = ABR_PINN_Orig(N_MODULES, F_BYZANTINE, EPSILON_BAR,
                         GAMMA_HIGH, GAMMA_LOW, W1, W2, physics_models)
    abr.reset()
    ewma_history = []
    for t in range(T_data):
        abr.step(soc_attacked[t], v_obs_all[t], i_obs_all[t],
                 t_obs_all[t], dt=dt_arr[t])
        ewma_history.append(abr.ewma_z.copy())

    ewma_history = np.array(ewma_history)
    t_ax = np.arange(T_data)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True,
                                    gridspec_kw={'height_ratios': [2, 1]})

    for j in range(N_MODULES):
        if j in byzantine_nodes:
            ax1.plot(t_ax, ewma_history[:, j], color='#e41a1c',
                     alpha=0.6, linewidth=1.0,
                     label='Byzantine' if j == byzantine_nodes[0] else None)
        else:
            ax1.plot(t_ax, ewma_history[:, j], color='#377eb8',
                     alpha=0.2, linewidth=0.6,
                     label='Honest' if j == 1 else None)

    ax1.axvline(x=t_start, color='gray', linestyle='--', linewidth=1,
                label='Attack onset')
    ax1.axhline(y=abr.MIN_EWMA_QUARANTINE, color='orange', linestyle=':',
                linewidth=1, label=f'Min quarantine ({abr.MIN_EWMA_QUARANTINE})')
    ax1.set_ylabel('EWMA z-score', fontsize=11)
    ax1.set_title('EWMA Z-Score Convergence (Constant Bias Attack, f=5)', fontsize=13)
    ax1.legend(fontsize=9, loc='upper left')
    ax1.grid(True, alpha=0.2)
    ax1.set_ylim(-0.5, abr.EWMA_CAP + 1)

    trust_history = np.array(abr.history_trust)
    for j in range(N_MODULES):
        if j in byzantine_nodes:
            ax2.plot(t_ax[:len(trust_history)], trust_history[:, j],
                     color='#e41a1c', alpha=0.6, linewidth=1.0)
        else:
            ax2.plot(t_ax[:len(trust_history)], trust_history[:, j],
                     color='#377eb8', alpha=0.2, linewidth=0.6)
    ax2.axvline(x=t_start, color='gray', linestyle='--', linewidth=1)
    ax2.axhline(y=GAMMA_LOW, color='gray', linestyle=':', linewidth=0.8,
                label=f'$\\gamma_{{low}}$ = {GAMMA_LOW}')
    ax2.set_ylabel('Trust $\\gamma_j$', fontsize=11)
    ax2.set_xlabel('Time step', fontsize=11)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.2)

    fig.tight_layout()
    path = OUTPUT_DIR / "fig_convergence.pdf"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"    Saved: {path}")


def generate_scalability_figure(scalability_results):
    """Grouped bar chart of FPR and TPR vs f."""
    print("\n  Generating scalability bar chart...")

    f_values = sorted(scalability_results.keys())
    methods_list = ['ABR-PINN', 'ABR-Stat', 'PBFT-BMS']
    colors = {'ABR-PINN': '#377eb8', 'ABR-Stat': '#984ea3', 'PBFT-BMS': '#a65628'}
    attack_types = ['constant_bias', 'random_noise', 'slow_drift', 'collusion']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    x = np.arange(len(f_values))
    width = 0.25

    for i, method in enumerate(methods_list):
        fpr_means = []
        tpr_means = []
        for f_val in f_values:
            fprs = [scalability_results[f_val][atk][method]['fpr_mean']
                    for atk in attack_types]
            tprs = [scalability_results[f_val][atk][method]['tpr_mean']
                    for atk in attack_types]
            fpr_means.append(np.mean(fprs))
            tpr_means.append(np.mean(tprs))

        ax1.bar(x + i * width, [f * 100 for f in fpr_means], width,
                label=method, color=colors[method], alpha=0.85)
        ax2.bar(x + i * width, [t * 100 for t in tpr_means], width,
                label=method, color=colors[method], alpha=0.85)

    for ax, ylabel, title in [
        (ax1, 'False Positive Rate (%)', 'FPR vs. Number of Byzantine Nodes'),
        (ax2, 'True Positive Rate (%)', 'TPR vs. Number of Byzantine Nodes')
    ]:
        ax.set_xlabel('Number of Byzantine nodes $f$', fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=12)
        ax.set_xticks(x + width)
        ax.set_xticklabels([str(f) for f in f_values])
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.2, axis='y')

    fig.tight_layout()
    path = OUTPUT_DIR / "fig3_scalability.pdf"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"    Saved: {path}")


# ===================================================================
# Main
# ===================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("ABR-PINN Sensitivity & Scalability Sweeps")
    print("=" * 60)

    print(f"\nLoading data from {DATA_FILE.name} ...")
    df = load_battery_data(DATA_FILE)
    print(f"  Loaded {len(df)} samples (subsample={SUBSAMPLE})")
    print(f"  Seeds per configuration: {SWEEP_SEEDS}")

    t_total = time.time()

    lambda_results = sweep_lambda_phys(df)
    w1_results = sweep_w1(df)
    scalability_results = sweep_scalability(df)

    print("\n" + "=" * 60)
    print("Generating publication figures...")
    print("=" * 60)
    generate_violin_figure(df)
    generate_convergence_figure(df)
    generate_scalability_figure(scalability_results)

    all_sweep = {
        'lambda_phys': {str(k): v for k, v in lambda_results.items()},
        'w1': {str(k): v for k, v in w1_results.items()},
        'scalability': {str(k): v for k, v in scalability_results.items()},
        'config': {
            'seeds': SWEEP_SEEDS,
            'subsample': SUBSAMPLE,
            'n_modules': N_MODULES,
        }
    }
    json_path = OUTPUT_DIR / "sensitivity_results.json"
    with open(json_path, 'w') as f:
        json.dump(all_sweep, f, indent=2, default=str)
    print(f"\nAll results saved to {json_path}")

    elapsed = time.time() - t_total
    print(f"\nTotal time: {elapsed:.0f}s ({elapsed/60:.1f} min)")
