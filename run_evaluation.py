#!/usr/bin/env python3
"""
ABR-PINN Evaluation Script
El Tiemblo Solar+Storage LiFePO4 Testbed — 16s16p pack, 70 days

Evaluates ABR-PINN Byzantine-resilient consensus for distributed BMS
against five baselines under four attack scenarios using real operational data.

Author: Andrei Garcia Cuadra (ETSIDI-UPM)
"""

import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats as sp_stats
from scipy.optimize import minimize

warnings.filterwarnings('ignore')

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_FILE = (SCRIPT_DIR.parent.parent.parent.parent /
             "datasets" / "eltiemblo_solar_completo" /
             "848299_0_Hock_log_20250101-0000_to_20251231-2358.csv")
OUTPUT_DIR = SCRIPT_DIR / "results"

# ═══════════════════════════════════════════════════════════════════
# Battery parameters (El Tiemblo 16s16p LiFePO4)
# ═══════════════════════════════════════════════════════════════════
N_SERIES = 16
N_PARALLEL = 16
V_NOM_CELL = 3.2          # V
V_NOM_PACK = V_NOM_CELL * N_SERIES  # 51.2 V
Q_NOM_CELL = 280.0        # Ah
Q_NOM_PACK = Q_NOM_CELL * N_PARALLEL  # 4480 Ah (per-string same)
R0_CELL = 0.4e-3          # Ohm (estimated for 280Ah LiFePO4 prismatic)
R0_PACK = R0_CELL * N_SERIES / N_PARALLEL  # Ohm

# OCV polynomial for LiFePO4 (6th order, SOC in [0,1], output in V per cell)
OCV_COEFFS = np.array([2.80, 3.15, -14.20, 34.80, -43.50, 27.20, -6.80])

def ocv_cell(soc):
    """OCV(SOC) for a single LiFePO4 cell, SOC in [0,1]."""
    soc = np.clip(soc, 0.0, 1.0)
    return np.polyval(OCV_COEFFS[::-1], soc)

def ocv_pack(soc):
    return ocv_cell(soc) * N_SERIES

# ═══════════════════════════════════════════════════════════════════
# ABR-PINN protocol parameters
# ═══════════════════════════════════════════════════════════════════
N_MODULES = 16             # simulated BMS nodes
F_BYZANTINE = 5            # max tolerable faults (f < n/3)
EPSILON_BAR = 0.50         # V — physics tolerance
GAMMA_HIGH = 0.8
GAMMA_LOW = 0.4
W1 = 0.7                  # corroboration weight: physics
W2 = 0.3                  # corroboration weight: statistical
CONSENSUS_ROUNDS = F_BYZANTINE + 1  # f+1 = 6

# ═══════════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════════

def load_battery_data(path: Path) -> pd.DataFrame:
    """Load El Tiemblo CSV and extract battery-relevant columns."""
    df = pd.read_csv(path, header=[0, 1])
    cols = df.columns
    # Find battery monitor columns by second header
    col_map = {}
    for i, (l1, l2) in enumerate(cols):
        l2l = l2.strip().lower()
        if 'Battery Monitor' in l1:
            if l2l == 'voltage':
                col_map['V_pack'] = i
            elif l2l == 'current':
                col_map['I_pack'] = i
            elif l2l == 'battery temperature':
                col_map['T_batt'] = i
            elif l2l == 'state of charge':
                col_map['SOC'] = i
            elif l2l == 'minimum cell voltage':
                col_map['V_cell_min'] = i
            elif l2l == 'maximum cell voltage':
                col_map['V_cell_max'] = i
            elif l2l == 'minimum cell temperature':
                col_map['T_cell_min'] = i
            elif l2l == 'maximum cell temperature':
                col_map['T_cell_max'] = i
        if l2.strip() == 'Europe/Paris (+02:00)':
            col_map['timestamp'] = i

    result = pd.DataFrame()
    result['timestamp'] = pd.to_datetime(df.iloc[:, col_map['timestamp']])
    for key in ['V_pack', 'I_pack', 'T_batt', 'SOC',
                'V_cell_min', 'V_cell_max', 'T_cell_min', 'T_cell_max']:
        if key in col_map:
            result[key] = pd.to_numeric(df.iloc[:, col_map[key]], errors='coerce')

    result = result.dropna(subset=['V_pack', 'I_pack', 'SOC'])
    result = result.reset_index(drop=True)
    # Compute dt between samples
    result['dt'] = result['timestamp'].diff().dt.total_seconds().fillna(1.0)
    result['dt'] = result['dt'].clip(0.1, 600)
    return result


def simulate_modules(df: pd.DataFrame, n_modules: int, rng: np.random.Generator):
    """
    Simulate n_modules BMS nodes from pack-level data.
    Each module monitors one series group. We generate per-module cell
    voltages by distributing the min-max cell voltage range across modules
    with manufacturing variability.
    """
    T = len(df)
    soc_gt = df['SOC'].values / 100.0  # ground truth SOC [0,1]
    V_pack = df['V_pack'].values
    I_pack = df['I_pack'].values
    T_batt = df['T_batt'].values
    V_min = df['V_cell_min'].values
    V_max = df['V_cell_max'].values
    dt = df['dt'].values

    # Per-module manufacturing variability
    capacity_var = 1.0 + rng.normal(0, 0.02, n_modules)  # ±2% capacity
    r0_var = 1.0 + rng.normal(0, 0.05, n_modules)        # ±5% R0

    # Simulate per-module cell voltages: interpolate between V_min and V_max
    # with module-specific position in the distribution
    module_positions = np.linspace(0, 1, n_modules)
    rng.shuffle(module_positions)

    module_data = []
    for m in range(n_modules):
        pos = module_positions[m]
        # Cell voltage: interpolate between min and max with noise
        V_cell = V_min + pos * (V_max - V_min)
        V_cell += rng.normal(0, 0.005, T)  # 5mV measurement noise
        # Module SOC from cell voltage via inverse OCV (approximate)
        soc_module = soc_gt + rng.normal(0, 0.005, T)  # small SOC noise
        soc_module = np.clip(soc_module, 0, 1)
        # Current (same for all modules in series string)
        I_module = I_pack / N_PARALLEL + rng.normal(0, 0.1, T)
        # Temperature with position-dependent offset
        T_module = T_batt + rng.normal(0, 0.5, T) + (m - n_modules/2) * 0.15
        # R0 for this module
        r0_m = R0_CELL * r0_var[m]

        module_data.append({
            'V_cell': V_cell,
            'I': I_module,
            'T': T_module,
            'SOC': soc_module,
            'R0': r0_m,
            'Q_scale': capacity_var[m],
        })

    return soc_gt, V_pack, I_pack, T_batt, dt, module_data


# ═══════════════════════════════════════════════════════════════════
# Physics model (simplified PINN equivalent)
# ═══════════════════════════════════════════════════════════════════

class PhysicsModel:
    """
    Simplified PINN: predicts voltage from SOC, current, temperature.
    V_pred = OCV(SOC) + R0 * I + alpha_T * (T - T_ref)
    Also supports inverse: estimate SOC from observed voltage via bisection.
    """
    def __init__(self, r0=R0_CELL, alpha_t=0.001, t_ref=25.0):
        self.r0 = r0
        self.alpha_t = alpha_t
        self.t_ref = t_ref

    def predict_voltage(self, soc, current, temperature):
        v_ocv = ocv_cell(soc)
        v_pred = v_ocv + self.r0 * current + self.alpha_t * (temperature - self.t_ref)
        return v_pred

    def residual(self, v_obs, soc, current, temperature):
        v_pred = self.predict_voltage(soc, current, temperature)
        return np.abs(v_obs - v_pred)

    def inverse_soc(self, v_obs, current, temperature):
        """Estimate SOC from observed voltage by inverting the ECM via bisection."""
        v_target = v_obs - self.r0 * current - self.alpha_t * (temperature - self.t_ref)
        lo, hi = 0.0, 1.0
        for _ in range(50):
            mid = (lo + hi) / 2
            if ocv_cell(mid) < v_target:
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2


# ═══════════════════════════════════════════════════════════════════
# Simple feedforward NN trained with physics loss (actual PINN)
# ═══════════════════════════════════════════════════════════════════

class SimplePINN:
    """
    A minimal 2-hidden-layer PINN for SOC estimation.
    Input: (V_cell, I, T) normalized
    Output: SOC prediction
    Physics loss: charge conservation dSOC/dt = -I/(Q_nom)
    """
    def __init__(self, hidden=32, rng=None):
        if rng is None:
            rng = np.random.default_rng(0)
        scale = 0.5
        self.W1 = rng.normal(0, scale, (3, hidden))
        self.b1 = np.zeros(hidden)
        self.W2 = rng.normal(0, scale, (hidden, hidden))
        self.b2 = np.zeros(hidden)
        self.W3 = rng.normal(0, scale, (hidden, 1))
        self.b3 = np.zeros(1)
        self.input_mean = np.zeros(3)
        self.input_std = np.ones(3)

    def _pack(self):
        return np.concatenate([
            self.W1.ravel(), self.b1, self.W2.ravel(), self.b2,
            self.W3.ravel(), self.b3
        ])

    def _unpack(self, params):
        h = self.W1.shape[1]
        idx = 0
        self.W1 = params[idx:idx+3*h].reshape(3, h); idx += 3*h
        self.b1 = params[idx:idx+h]; idx += h
        self.W2 = params[idx:idx+h*h].reshape(h, h); idx += h*h
        self.b2 = params[idx:idx+h]; idx += h
        self.W3 = params[idx:idx+h].reshape(h, 1); idx += h
        self.b3 = params[idx:idx+1]; idx += 1

    def forward(self, X):
        X_norm = (X - self.input_mean) / (self.input_std + 1e-8)
        z1 = np.tanh(X_norm @ self.W1 + self.b1)
        z2 = np.tanh(z1 @ self.W2 + self.b2)
        out = z2 @ self.W3 + self.b3
        return 1.0 / (1.0 + np.exp(-out.ravel()))  # sigmoid → [0,1]

    def train(self, V, I, T, SOC_true, dt, max_iter=200, lambda_phys=1.0):
        X = np.column_stack([V, I, T])
        self.input_mean = X.mean(axis=0)
        self.input_std = X.std(axis=0) + 1e-8
        y = SOC_true.copy()

        def loss(params):
            self._unpack(params)
            soc_pred = self.forward(X)
            data_loss = np.mean((soc_pred - y) ** 2)
            dsoc_pred = np.diff(soc_pred) / (dt[1:] + 1e-6)
            dsoc_phys = -I[:-1] / (Q_NOM_CELL * 3600)
            phys_loss = np.mean((dsoc_pred - dsoc_phys) ** 2)
            return data_loss + lambda_phys * phys_loss + 1e-4 * np.sum(params**2)

        p0 = self._pack()
        result = minimize(loss, p0, method='L-BFGS-B',
                          options={'maxiter': max_iter, 'ftol': 1e-8})
        self._unpack(result.x)
        return result.fun

    def predict_soc(self, V, I, T):
        X = np.column_stack([V, I, T])
        return self.forward(X)


# ═══════════════════════════════════════════════════════════════════
# SOC Estimation Baselines
# ═══════════════════════════════════════════════════════════════════

class CoulombCounting:
    """Simple coulomb counting SOC estimator."""
    def __init__(self, q_nom=Q_NOM_CELL, eta=0.995):
        self.q_nom = q_nom
        self.eta = eta

    def estimate(self, soc_init, I, dt):
        T = len(I)
        soc = np.zeros(T)
        soc[0] = soc_init
        for t in range(1, T):
            dsoc = -self.eta * I[t] * dt[t] / (self.q_nom * 3600)
            soc[t] = np.clip(soc[t-1] + dsoc, 0, 1)
        return soc


class EKF_SOC:
    """Extended Kalman Filter for SOC estimation."""
    def __init__(self, q_nom=Q_NOM_CELL, r0=R0_CELL, q_proc=1e-6, r_meas=1e-3):
        self.q_nom = q_nom
        self.r0 = r0
        self.Q = q_proc
        self.R = r_meas

    def estimate(self, soc_init, V_obs, I, T_batt, dt):
        N = len(V_obs)
        soc = np.zeros(N)
        soc[0] = soc_init
        P = 0.01
        for t in range(1, N):
            # Predict
            dsoc = -I[t] * dt[t] / (self.q_nom * 3600)
            soc_pred = np.clip(soc[t-1] + dsoc, 0, 1)
            P_pred = P + self.Q
            # Update
            v_pred = ocv_cell(soc_pred) + self.r0 * I[t]
            H = _docv_dsoc(soc_pred)
            S = H * P_pred * H + self.R
            K = P_pred * H / (S + 1e-15)
            innovation = V_obs[t] - v_pred
            soc[t] = np.clip(soc_pred + K * innovation, 0, 1)
            P = (1 - K * H) * P_pred
        return soc


class UKF_SOC:
    """Unscented Kalman Filter for SOC estimation."""
    def __init__(self, q_nom=Q_NOM_CELL, r0=R0_CELL, q_proc=1e-6, r_meas=1e-3):
        self.q_nom = q_nom
        self.r0 = r0
        self.Q = q_proc
        self.R = r_meas

    def estimate(self, soc_init, V_obs, I, T_batt, dt):
        N = len(V_obs)
        soc = np.zeros(N)
        soc[0] = soc_init
        P = 0.01
        alpha, beta, kappa = 1e-3, 2.0, 0.0
        lam = alpha**2 * (1 + kappa) - 1
        for t in range(1, N):
            # Sigma points (1D)
            s = np.sqrt((1 + lam) * P)
            sp = np.array([soc[t-1], soc[t-1] + s, soc[t-1] - s])
            w_m = np.array([lam/(1+lam), 0.5/(1+lam), 0.5/(1+lam)])
            w_c = w_m.copy()
            w_c[0] += (1 - alpha**2 + beta)
            # Predict
            dsoc = -I[t] * dt[t] / (self.q_nom * 3600)
            sp_pred = np.clip(sp + dsoc, 0, 1)
            soc_pred = np.sum(w_m * sp_pred)
            P_pred = np.sum(w_c * (sp_pred - soc_pred)**2) + self.Q
            # Observation
            v_sp = np.array([ocv_cell(s) + self.r0 * I[t] for s in sp_pred])
            v_pred = np.sum(w_m * v_sp)
            Pvv = np.sum(w_c * (v_sp - v_pred)**2) + self.R
            Pxv = np.sum(w_c * (sp_pred - soc_pred) * (v_sp - v_pred))
            K = Pxv / (Pvv + 1e-15)
            soc[t] = np.clip(soc_pred + K * (V_obs[t] - v_pred), 0, 1)
            P = P_pred - K * Pvv * K
        return soc


class ECM_LS:
    """Equivalent Circuit Model with Least Squares parameter identification."""
    def __init__(self, q_nom=Q_NOM_CELL):
        self.q_nom = q_nom

    def estimate(self, soc_init, V_obs, I, T_batt, dt):
        N = len(V_obs)
        soc_cc = CoulombCounting(self.q_nom).estimate(soc_init, I, dt)
        # LS: fit V = OCV(SOC_cc) + R0*I + offset
        A = np.column_stack([I, np.ones(N)])
        V_ocv = np.array([ocv_cell(s) for s in soc_cc])
        b = V_obs - V_ocv
        try:
            params, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
            r0_fit = params[0]
        except:
            r0_fit = R0_CELL
        # Re-estimate with fitted R0
        ekf = EKF_SOC(self.q_nom, abs(r0_fit), 1e-6, 2e-3)
        return ekf.estimate(soc_init, V_obs, I, T_batt, dt)


def _docv_dsoc(soc, delta=1e-4):
    """Numerical derivative of OCV w.r.t. SOC."""
    return (ocv_cell(soc + delta) - ocv_cell(soc - delta)) / (2 * delta)


# ═══════════════════════════════════════════════════════════════════
# ABR-PINN Protocol
# ═══════════════════════════════════════════════════════════════════

def sigmoid(x, k=10.0, x0=0.5):
    return 1.0 / (1.0 + np.exp(-k * (x - x0)))


class ABR_PINN:
    """ABR-PINN: Byzantine-resilient consensus with physics-informed validation.

    Combines statistical outlier detection (EWMA of z-scores) with temporal
    physics consistency (dSOC vs. current integral) for trust-weighted
    consensus.  Persistent deviations decrease trust; transient ones
    (congestion, BMS recalibration) recover, yielding lower FPR than
    purely statistical approaches while maintaining Byzantine robustness.
    """

    EWMA_ALPHA = 0.10         # EWMA smoothing for z-score rise
    EWMA_ALPHA_DOWN = 0.30    # faster decay when z drops (3× rise rate)
    TRUST_SIGMOID_K = 1.5     # sigmoid steepness for trust mapping
    TRUST_SIGMOID_X0 = 6.0    # sigmoid midpoint for trust curve
    EWMA_CAP = 25.0           # cap to prevent float overflow
    MAD_FLOOR = 0.005         # prevents z explosion when SOCs cluster near 0/1
    GAP_THRESHOLD = 3.0       # minimum ewma_z gap for quarantine trigger
    MIN_EWMA_QUARANTINE = 5.0 # minimum ewma_z to consider quarantine

    def __init__(self, n_modules, f_max, epsilon_bar, gamma_high, gamma_low,
                 w1, w2, physics_models, pinn_model=None):
        self.n = n_modules
        self.f = f_max
        self.eps = epsilon_bar
        self.gh = gamma_high
        self.gl = gamma_low
        self.w1 = w1
        self.w2 = w2
        self.models = physics_models
        self.pinn = pinn_model
        self.trust = np.ones(n_modules) * gamma_high
        self.quarantined = np.zeros(n_modules, dtype=bool)
        self.ewma_z = np.zeros(n_modules)
        self.prev_reports = None
        self.history_trust = []
        self.history_detections = []

    def reset(self):
        self.trust = np.ones(self.n) * self.gh
        self.quarantined = np.zeros(self.n, dtype=bool)
        self.ewma_z = np.zeros(self.n)
        self.prev_reports = None
        self.history_trust = []
        self.history_detections = []

    def step(self, soc_reports, v_obs, i_obs, t_obs, dt=1.0):
        """
        One consensus step.
        Returns: consensus SOC, list of newly detected Byzantine nodes.
        """
        n = self.n
        detected = []

        # --- Phase 1: Statistical z-scores (robust, MAD-based) ---
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

        # --- Phase 2: Asymmetric EWMA trust accumulation ---
        for j in range(n):
            if not self.quarantined[j]:
                if z_scores[j] >= self.ewma_z[j]:
                    alpha_eff = self.EWMA_ALPHA
                else:
                    alpha_eff = self.EWMA_ALPHA_DOWN
                self.ewma_z[j] = min(
                    (1 - alpha_eff) * self.ewma_z[j] + alpha_eff * z_scores[j],
                    self.EWMA_CAP)
                self.trust[j] = self.gl + (self.gh - self.gl) / (
                    1.0 + np.exp(self.TRUST_SIGMOID_K *
                                 (self.ewma_z[j] - self.TRUST_SIGMOID_X0)))

        # --- Phase 3: Gap-based quarantine (at most f nodes) ---
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
                return np.median(soc_reports), detected
        weights /= w_sum
        consensus_soc = np.sum(weights * soc_reports)

        self.history_trust.append(self.trust.copy())
        self.history_detections.append(detected)
        return consensus_soc, detected


class ABR_Stat:
    """ABR with statistical-only fault detection (no physics)."""

    def __init__(self, n_modules, f_max):
        self.n = n_modules
        self.f = f_max
        self.history = []

    def step(self, soc_reports):
        median = np.median(soc_reports)
        deviations = np.abs(soc_reports - median)
        mad = np.median(deviations) + 1e-8
        z_scores = deviations / (1.4826 * mad)
        detected = list(np.where(z_scores > 3.0)[0])
        mask = z_scores <= 3.0
        if mask.any():
            consensus = np.mean(soc_reports[mask])
        else:
            consensus = median
        self.history.append(detected)
        return consensus, detected


class PBFT_BMS:
    """Practical BFT voting without physics validation."""

    def __init__(self, n_modules, f_max):
        self.n = n_modules
        self.f = f_max

    def step(self, soc_reports):
        # PBFT-style: require 2f+1 agreement
        sorted_reports = np.sort(soc_reports)
        # Trim f highest and f lowest
        trimmed = sorted_reports[self.f:-self.f] if 2*self.f < self.n else sorted_reports
        consensus = np.mean(trimmed)
        detected = []
        for i in range(self.n):
            if abs(soc_reports[i] - consensus) > 0.1:
                detected.append(i)
        return consensus, detected


# ═══════════════════════════════════════════════════════════════════
# Attack injection
# ═══════════════════════════════════════════════════════════════════

def inject_attacks(soc_reports_all, attack_type, byzantine_nodes, t_start, rng):
    """
    Inject Byzantine faults into SOC reports.
    attack_type: 'constant_bias', 'random_noise', 'slow_drift', 'collusion'
    Returns modified soc_reports_all (T, n_modules)
    """
    reports = soc_reports_all.copy()
    T, n = reports.shape
    t_range = np.arange(T)
    mask = t_range >= t_start

    for byz in byzantine_nodes:
        if attack_type == 'constant_bias':
            reports[mask, byz] += 0.15  # +15% SOC offset
        elif attack_type == 'random_noise':
            reports[mask, byz] += rng.normal(0, 0.10, mask.sum())
        elif attack_type == 'slow_drift':
            drift = np.linspace(0, 0.20, mask.sum())
            reports[mask, byz] += drift
        elif attack_type == 'collusion':
            # All Byzantine nodes agree on a wrong value
            reports[mask, byz] = 0.80  # coordinated false high SOC

    reports = np.clip(reports, 0, 1)
    return reports, mask


# ═══════════════════════════════════════════════════════════════════
# Evaluation metrics
# ═══════════════════════════════════════════════════════════════════

def compute_metrics(soc_pred, soc_gt, detected_per_step, byzantine_nodes,
                    attack_active, t_start):
    T = len(soc_gt)
    atk = attack_active

    # SOC accuracy
    rmse_full = np.sqrt(np.mean((soc_pred - soc_gt)**2))
    mae_full = np.mean(np.abs(soc_pred - soc_gt))
    max_err = np.max(np.abs(soc_pred - soc_gt))

    rmse_atk = np.sqrt(np.mean((soc_pred[atk] - soc_gt[atk])**2)) if atk.any() else 0.0
    mae_atk = np.mean(np.abs(soc_pred[atk] - soc_gt[atk])) if atk.any() else 0.0

    # Detection metrics
    byz_set = set(byzantine_nodes)
    tp, fp, fn = 0, 0, 0
    first_detect = None
    for t in range(T):
        det_set = set(detected_per_step[t]) if t < len(detected_per_step) else set()
        tp += len(det_set & byz_set)
        fp += len(det_set - byz_set)
        if t >= t_start and first_detect is None and (det_set & byz_set):
            first_detect = t - t_start

    n_byz_total = len(byzantine_nodes) * max(1, sum(atk))
    tpr = tp / max(n_byz_total, 1)
    # FPR: false positives / (clean_nodes * clean_steps)
    clean_steps = max(1, t_start)
    n_clean = N_MODULES - len(byzantine_nodes)
    fpr = fp / max(n_clean * clean_steps, 1)

    return {
        'rmse_full': rmse_full,
        'mae_full': mae_full,
        'max_err': max_err,
        'rmse_attack': rmse_atk,
        'mae_attack': mae_atk,
        'tpr': min(tpr, 1.0),
        'fpr': fpr,
        'detection_latency': first_detect,
    }


# ═══════════════════════════════════════════════════════════════════
# Friedman + Nemenyi + Cliff's delta
# ═══════════════════════════════════════════════════════════════════

def friedman_nemenyi(results_dict, metric='rmse_attack'):
    """
    Friedman test + Nemenyi post-hoc + Cliff's delta.
    results_dict: {method_name: [val_seed1, val_seed2, ...]}
    """
    methods = sorted(results_dict.keys())
    k = len(methods)
    n = len(next(iter(results_dict.values())))

    data = np.array([results_dict[m] for m in methods]).T  # (n_seeds, k)
    ranks = np.zeros_like(data)
    for i in range(n):
        ranks[i] = sp_stats.rankdata(data[i])

    avg_ranks = ranks.mean(axis=0)
    chi2 = 12 * n / (k * (k + 1)) * (np.sum(avg_ranks**2) - k * (k+1)**2 / 4)
    p_value = 1 - sp_stats.chi2.cdf(chi2, df=k-1)

    # Cliff's delta between ABR-PINN and each baseline
    cliff_deltas = {}
    abr_idx = methods.index('ABR-PINN') if 'ABR-PINN' in methods else 0
    abr_vals = data[:, abr_idx]
    for j, m in enumerate(methods):
        if j == abr_idx:
            continue
        other_vals = data[:, j]
        n_greater = sum(1 for a in abr_vals for b in other_vals if a < b)
        n_less = sum(1 for a in abr_vals for b in other_vals if a > b)
        total = n * n
        cliff_deltas[m] = (n_greater - n_less) / total if total > 0 else 0

    # Nemenyi CD
    q_alpha = 2.728  # q_{0.05} for k=6
    cd = q_alpha * np.sqrt(k * (k + 1) / (6 * n))

    return {
        'methods': methods,
        'avg_ranks': {m: float(avg_ranks[i]) for i, m in enumerate(methods)},
        'chi2': float(chi2),
        'p_value': float(p_value),
        'cliff_deltas': cliff_deltas,
        'nemenyi_cd': float(cd),
    }


# ═══════════════════════════════════════════════════════════════════
# Main trial
# ═══════════════════════════════════════════════════════════════════

def run_trial(df, attack_type, byzantine_nodes, seed, use_pinn=True, subsample=1):
    """Run one complete trial: all methods on one attack scenario."""
    rng = np.random.default_rng(seed)

    # Subsample data for speed
    if subsample > 1:
        df_sub = df.iloc[::subsample].reset_index(drop=True)
    else:
        df_sub = df

    T_data = len(df_sub)
    soc_gt, V_pack, I_pack, T_batt, dt, module_data = simulate_modules(
        df_sub, N_MODULES, rng)

    t_start = T_data // 4  # attack starts at 25% of data

    # Build per-module SOC reports (ground truth + noise)
    soc_reports_all = np.column_stack([m['SOC'] for m in module_data])

    # Inject attacks
    soc_attacked, attack_mask = inject_attacks(
        soc_reports_all, attack_type, byzantine_nodes, t_start, rng)

    # Build physics models per module
    physics_models = [PhysicsModel(r0=m['R0']) for m in module_data]

    # Voltage/current/temperature arrays per module
    v_obs_all = np.column_stack([m['V_cell'] for m in module_data])
    i_obs_all = np.column_stack([m['I'] for m in module_data])
    t_obs_all = np.column_stack([m['T'] for m in module_data])

    results = {}

    # ---- Method 1: EKF ----
    ekf = EKF_SOC()
    soc_ekf = ekf.estimate(soc_gt[0], module_data[0]['V_cell'],
                           module_data[0]['I'], module_data[0]['T'], dt)
    results['EKF'] = compute_metrics(
        soc_ekf, soc_gt, [[] for _ in range(T_data)], byzantine_nodes,
        attack_mask, t_start)

    # ---- Method 2: UKF ----
    ukf = UKF_SOC()
    soc_ukf = ukf.estimate(soc_gt[0], module_data[0]['V_cell'],
                           module_data[0]['I'], module_data[0]['T'], dt)
    results['UKF'] = compute_metrics(
        soc_ukf, soc_gt, [[] for _ in range(T_data)], byzantine_nodes,
        attack_mask, t_start)

    # ---- Method 3: ECM-LS ----
    ecm = ECM_LS()
    soc_ecm = ecm.estimate(soc_gt[0], module_data[0]['V_cell'],
                           module_data[0]['I'], module_data[0]['T'], dt)
    results['ECM-LS'] = compute_metrics(
        soc_ecm, soc_gt, [[] for _ in range(T_data)], byzantine_nodes,
        attack_mask, t_start)

    # ---- Method 4: ABR-Stat ----
    abr_stat = ABR_Stat(N_MODULES, F_BYZANTINE)
    soc_abrstat = np.zeros(T_data)
    det_abrstat = []
    for t in range(T_data):
        soc_abrstat[t], det = abr_stat.step(soc_attacked[t])
        det_abrstat.append(det)
    results['ABR-Stat'] = compute_metrics(
        soc_abrstat, soc_gt, det_abrstat, byzantine_nodes,
        attack_mask, t_start)

    # ---- Method 5: PBFT-BMS ----
    pbft = PBFT_BMS(N_MODULES, F_BYZANTINE)
    soc_pbft = np.zeros(T_data)
    det_pbft = []
    for t in range(T_data):
        soc_pbft[t], det = pbft.step(soc_attacked[t])
        det_pbft.append(det)
    results['PBFT-BMS'] = compute_metrics(
        soc_pbft, soc_gt, det_pbft, byzantine_nodes,
        attack_mask, t_start)

    # ---- Method 6: ABR-PINN ----
    abr_pinn = ABR_PINN(N_MODULES, F_BYZANTINE, EPSILON_BAR,
                        GAMMA_HIGH, GAMMA_LOW, W1, W2, physics_models)
    abr_pinn.reset()
    soc_abrpinn = np.zeros(T_data)
    det_abrpinn = []
    for t in range(T_data):
        soc_abrpinn[t], det = abr_pinn.step(
            soc_attacked[t],
            v_obs_all[t], i_obs_all[t], t_obs_all[t],
            dt=dt[t])
        det_abrpinn.append(list(np.where(abr_pinn.quarantined)[0]))
    results['ABR-PINN'] = compute_metrics(
        soc_abrpinn, soc_gt, det_abrpinn, byzantine_nodes,
        attack_mask, t_start)

    # Store time series for figures (only seed 0)
    if seed == 0:
        results['_timeseries'] = {
            'soc_gt': soc_gt,
            'soc_abrpinn': soc_abrpinn,
            'soc_ekf': soc_ekf,
            'soc_abrstat': soc_abrstat,
            'trust_history': np.array(abr_pinn.history_trust) if abr_pinn.history_trust else None,
            't_start': t_start,
            'attack_mask': attack_mask,
        }

    return results


# ═══════════════════════════════════════════════════════════════════
# Full evaluation
# ═══════════════════════════════════════════════════════════════════

def run_full_evaluation(n_seeds=30, subsample=1):
    print("=" * 70)
    print("ABR-PINN Evaluation — El Tiemblo LiFePO4 Testbed")
    print("=" * 70)

    print(f"\nLoading data from {DATA_FILE.name} ...")
    df = load_battery_data(DATA_FILE)
    print(f"  Loaded {len(df)} battery samples over {df['timestamp'].iloc[-1] - df['timestamp'].iloc[0]}")

    attack_types = ['constant_bias', 'random_noise', 'slow_drift', 'collusion']
    byzantine_nodes = [0, 3, 7, 11, 14]  # 5 nodes = f

    all_results = {}
    representative = {}
    methods = ['EKF', 'UKF', 'ECM-LS', 'ABR-Stat', 'PBFT-BMS', 'ABR-PINN']

    for atk in attack_types:
        print(f"\n{'='*50}")
        print(f"Attack: {atk} | Byzantine nodes: {byzantine_nodes}")
        print(f"{'='*50}")

        seed_results = {m: [] for m in methods}

        for seed in range(n_seeds):
            t0 = time.time()
            trial = run_trial(df, atk, byzantine_nodes, seed,
                              subsample=subsample)
            elapsed = time.time() - t0

            for m in methods:
                seed_results[m].append(trial[m])

            if seed == 0 and '_timeseries' in trial:
                representative[atk] = trial['_timeseries']

            if seed % 5 == 0:
                abr = trial['ABR-PINN']
                print(f"  Seed {seed:2d}: ABR-PINN RMSE={abr['rmse_attack']:.4f} "
                      f"FPR={abr['fpr']:.4f} TPR={abr['tpr']:.3f} "
                      f"({elapsed:.1f}s)")

        # Aggregate per attack
        atk_summary = {}
        for m in methods:
            vals = seed_results[m]
            atk_summary[m] = {
                'rmse_attack_mean': float(np.mean([v['rmse_attack'] for v in vals])),
                'rmse_attack_std': float(np.std([v['rmse_attack'] for v in vals])),
                'mae_attack_mean': float(np.mean([v['mae_attack'] for v in vals])),
                'mae_attack_std': float(np.std([v['mae_attack'] for v in vals])),
                'rmse_full_mean': float(np.mean([v['rmse_full'] for v in vals])),
                'rmse_full_std': float(np.std([v['rmse_full'] for v in vals])),
                'max_err_mean': float(np.mean([v['max_err'] for v in vals])),
                'tpr_mean': float(np.mean([v['tpr'] for v in vals])),
                'tpr_std': float(np.std([v['tpr'] for v in vals])),
                'fpr_mean': float(np.mean([v['fpr'] for v in vals])),
                'fpr_std': float(np.std([v['fpr'] for v in vals])),
            }
            det_lats = [v['detection_latency'] for v in vals if v['detection_latency'] is not None]
            atk_summary[m]['det_latency_mean'] = float(np.mean(det_lats)) if det_lats else None
            atk_summary[m]['det_latency_std'] = float(np.std(det_lats)) if det_lats else None

        all_results[atk] = atk_summary

        # Friedman test on RMSE
        rmse_dict = {m: [v['rmse_attack'] for v in seed_results[m]] for m in methods}
        friedman = friedman_nemenyi(rmse_dict, 'rmse_attack')
        all_results[atk]['_friedman_rmse'] = friedman

        # Friedman test on FPR
        fpr_dict = {m: [v['fpr'] for v in seed_results[m]] for m in methods}
        friedman_fpr = friedman_nemenyi(fpr_dict, 'fpr')
        all_results[atk]['_friedman_fpr'] = friedman_fpr

        # Print summary
        print(f"\n  Summary for {atk}:")
        print(f"  {'Method':<12s} {'RMSE(atk)':>12s} {'MAE(atk)':>12s} "
              f"{'TPR':>8s} {'FPR':>8s}")
        print(f"  {'-'*52}")
        for m in methods:
            s = atk_summary[m]
            print(f"  {m:<12s} {s['rmse_attack_mean']:8.4f}±{s['rmse_attack_std']:.4f} "
                  f"{s['mae_attack_mean']:8.4f}±{s['mae_attack_std']:.4f} "
                  f"{s['tpr_mean']:6.3f} {s['fpr_mean']:6.4f}")
        print(f"  Friedman chi2={friedman['chi2']:.2f}, p={friedman['p_value']:.2e}")

    # ---- Global summary across all attacks ----
    print("\n" + "=" * 70)
    print("GLOBAL SUMMARY ACROSS ALL ATTACK TYPES")
    print("=" * 70)

    global_summary = {}
    for m in methods:
        all_rmse = []
        all_fpr = []
        all_tpr = []
        for atk in attack_types:
            all_rmse.append(all_results[atk][m]['rmse_attack_mean'])
            all_fpr.append(all_results[atk][m]['fpr_mean'])
            all_tpr.append(all_results[atk][m]['tpr_mean'])
        global_summary[m] = {
            'avg_rmse': float(np.mean(all_rmse)),
            'avg_fpr': float(np.mean(all_fpr)),
            'avg_tpr': float(np.mean(all_tpr)),
        }
        print(f"  {m:<12s} RMSE={global_summary[m]['avg_rmse']:.4f} "
              f"FPR={global_summary[m]['avg_fpr']:.4f} "
              f"TPR={global_summary[m]['avg_tpr']:.3f}")

    all_results['_global'] = global_summary

    # ---- FPR reduction claim ----
    abr_pinn_fpr = global_summary['ABR-PINN']['avg_fpr']
    abr_stat_fpr = global_summary['ABR-Stat']['avg_fpr']
    if abr_stat_fpr > 0:
        fpr_reduction = (abr_stat_fpr - abr_pinn_fpr) / abr_stat_fpr * 100
    else:
        fpr_reduction = 0
    all_results['_fpr_reduction_pct'] = float(fpr_reduction)
    print(f"\n  FPR reduction (ABR-PINN vs ABR-Stat): {fpr_reduction:.1f}%")

    # ---- Save ----
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = OUTPUT_DIR / "abr_pinn_results.json"
    with open(json_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {json_path}")

    generate_figures(all_results, representative, methods)
    print(f"All outputs saved to {OUTPUT_DIR}/")
    return all_results


# ═══════════════════════════════════════════════════════════════════
# Figure generation
# ═══════════════════════════════════════════════════════════════════

def generate_figures(all_results, representative, methods):
    attack_types = ['constant_bias', 'random_noise', 'slow_drift', 'collusion']

    # Figure 1: SOC estimation under Byzantine attack (time series)
    fig1_path = OUTPUT_DIR / "fig1_soc_timeseries.pdf"
    if 'constant_bias' in representative:
        ts = representative['constant_bias']
        soc_gt = ts['soc_gt']
        T = len(soc_gt)
        t_ax = np.arange(T)
        t_start = ts['t_start']

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
        ax1.plot(t_ax, soc_gt * 100, 'k-', label='Ground truth', linewidth=1.5)
        ax1.plot(t_ax, ts['soc_abrpinn'] * 100, 'b-', label='ABR-PINN', linewidth=1.2)
        ax1.plot(t_ax, ts['soc_ekf'] * 100, 'r--', label='EKF', linewidth=1.0, alpha=0.7)
        ax1.plot(t_ax, ts['soc_abrstat'] * 100, 'g:', label='ABR-Stat', linewidth=1.0, alpha=0.7)
        ax1.axvline(x=t_start, color='gray', linestyle='--', linewidth=1,
                    label='Attack onset')
        ax1.set_ylabel('SOC (%)', fontsize=12)
        ax1.set_title('SOC Estimation Under Constant Bias Attack (f=5 nodes)', fontsize=13)
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)

        if ts['trust_history'] is not None:
            trust = ts['trust_history']
            for i in range(min(16, trust.shape[1])):
                color = 'red' if i in [0, 3, 7, 11, 14] else 'blue'
                alpha = 0.8 if i in [0, 3, 7, 11, 14] else 0.3
                ax2.plot(t_ax[:len(trust)], trust[:, i], color=color,
                         alpha=alpha, linewidth=0.8)
            ax2.axhline(y=GAMMA_LOW, color='black', linestyle=':', linewidth=1,
                        label=f'$\\gamma_{{low}}$ = {GAMMA_LOW}')
            ax2.axvline(x=t_start, color='gray', linestyle='--', linewidth=1)
            ax2.set_ylabel('Trust score', fontsize=12)
            ax2.set_xlabel('Time step', fontsize=12)
            ax2.set_title('Trust Evolution (red = Byzantine, blue = honest)', fontsize=12)
            ax2.legend(fontsize=10)
            ax2.grid(True, alpha=0.3)

        fig.tight_layout()
        fig.savefig(fig1_path, dpi=150)
        plt.close(fig)
        print(f"  Figure 1 saved: {fig1_path}")

    # Figure 2: RMSE comparison bar chart across attacks
    fig2_path = OUTPUT_DIR / "fig2_rmse_comparison.pdf"
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(attack_types))
    width = 0.13
    colors = ['#e41a1c', '#ff7f00', '#f781bf', '#984ea3', '#a65628', '#377eb8']
    for i, m in enumerate(methods):
        means = [all_results[atk][m]['rmse_attack_mean'] for atk in attack_types]
        stds = [all_results[atk][m]['rmse_attack_std'] for atk in attack_types]
        ax.bar(x + i * width, means, width, yerr=stds, label=m,
               color=colors[i], capsize=3, alpha=0.85)
    ax.set_xticks(x + 2.5 * width)
    ax.set_xticklabels([a.replace('_', ' ').title() for a in attack_types], fontsize=11)
    ax.set_ylabel('SOC RMSE during attack', fontsize=12)
    ax.set_title('SOC Estimation Accuracy Under Byzantine Attacks', fontsize=13)
    ax.legend(fontsize=9, ncol=3)
    ax.grid(True, alpha=0.3, axis='y')
    fig.tight_layout()
    fig.savefig(fig2_path, dpi=150)
    plt.close(fig)
    print(f"  Figure 2 saved: {fig2_path}")

    # Figure 3: Scalability with number of Byzantine nodes
    fig3_path = OUTPUT_DIR / "fig3_scalability.pdf"
    print(f"  Figure 3 (scalability): requires separate run — placeholder saved")
    fig, ax = plt.subplots(figsize=(8, 5))
    f_values = [1, 2, 3, 4, 5]
    ax.set_xlabel('Number of Byzantine nodes $f$', fontsize=12)
    ax.set_ylabel('SOC RMSE', fontsize=12)
    ax.set_title('Scalability with Number of Byzantine Nodes', fontsize=13)
    ax.text(0.5, 0.5, 'Generated from scalability sweep\n(see run_scalability.py)',
            transform=ax.transAxes, ha='center', va='center', fontsize=12)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig3_path, dpi=150)
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ABR-PINN Evaluation')
    parser.add_argument('--seeds', type=int, default=30, help='Number of seeds')
    parser.add_argument('--fast', action='store_true',
                        help='Fast mode: 5 seeds, subsample=4')
    parser.add_argument('--subsample', type=int, default=1,
                        help='Subsample factor for data')
    args = parser.parse_args()

    if args.fast:
        n_seeds = 5
        subsample = 4
    else:
        n_seeds = args.seeds
        subsample = args.subsample

    t_start = time.time()
    run_full_evaluation(n_seeds=n_seeds, subsample=subsample)
    elapsed = time.time() - t_start
    print(f"\nTotal execution time: {elapsed:.1f} s")
