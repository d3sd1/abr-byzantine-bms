#!/usr/bin/env python3
"""
signal_model.py -- v3 per-module signal reconstruction for the El Tiemblo pack.

Replaces `run_evaluation.simulate_modules`, which set every module SOC to
SOC_pack + N(0, 0.005) and therefore made the 16 nodes near-copies of each
other (Applied Energy R1.2).  Here the 16 series groups differ because of
physically grounded quantities -- capacity spread, ohmic spread, initial
imbalance and current-sensor calibration -- and every node runs its own
honest local estimator, so the spread between reports is an OUTPUT of the
model, not an input noise term.

Signal chain per series group m (m = 0..15):

  true state       s_m(t)  = SOC_bms(t) + e_m + d_m(t)   [+ leak, if faulted]
  true voltage     V_m(t)  anchored inside the MEASURED [V_min, V_max]
                           envelope, ordered by the module's own OCV
  measured signals V^_m = V_m + b_V,m + n_V
                   I^_m = I_pack + b_I,m + n_I
                   T^_m = T_batt + grad*(m-7.5) + n_T
  reported SOC     s^_m(t) = local Coulomb counter on I^_m with the NOMINAL
                             capacity + OCV re-anchoring at the knees at rest
                             + full-charge reset  (what a JK BMS actually does)

CURRENT SIGN CONVENTION -- verified on the data, not assumed:
    POSITIVE current = CHARGING.
    Evidence: mean I_pack = +175.7 A over the samples where the BMS SOC
    increases and -56.2 A where it decreases; the windowed regression of
    Delta SOC on integral(I dt) has a positive slope for windows of 50 and
    200 samples.  This is the usual Victron convention.
    (run_evaluation.py, round 1, integrated with the opposite sign.)

Author: Andrei Garcia Cuadra (ETSIDI-UPM)
"""

from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_FILE = (SCRIPT_DIR.parent.parent.parent.parent /
             "datasets" / "eltiemblo_solar_completo" /
             "848299_0_Hock_log_20250101-0000_to_20251231-2358.csv")

# ------------------------------------------------------------------ pack
N_MODULES = 16          # series groups, one BMS node each
N_PARALLEL = 16         # cells in parallel inside a group
Q_NOM_CELL = 280.0      # Ah
Q_NOM_GROUP = Q_NOM_CELL * N_PARALLEL   # 4480 Ah -- declared 16s16p topology
R0_CELL = 0.4e-3        # Ohm
R0_GROUP = R0_CELL / N_PARALLEL

OCV_COEFFS = np.array([2.80, 3.15, -14.20, 34.80, -43.50, 27.20, -6.80])


def ocv_cell(soc):
    """OCV(SOC) of one LiFePO4 cell, SOC in [0,1] -> V."""
    return np.polyval(OCV_COEFFS[::-1], np.clip(soc, 0.0, 1.0))


_S_GRID = np.linspace(0.0, 1.0, 4001)
_V_GRID = ocv_cell(_S_GRID)
assert np.all(np.diff(_V_GRID) > 0), "OCV polynomial must be monotonic on [0,1]"


def inv_ocv(v):
    """Inverse OCV by monotone interpolation (vectorised)."""
    return np.interp(v, _V_GRID, _S_GRID)


def docv_dsoc(soc, delta=1e-4):
    return (ocv_cell(soc + delta) - ocv_cell(soc - delta)) / (2 * delta)


# --------------------------------------------------------- default config
DEFAULTS = dict(
    sigma_Q=0.02, sigma_R0=0.05, sigma_e0=0.01,
    sigma_bias_V=0.003, sigma_noise_V=0.002,
    I_FS=500.0, sigma_bias_I_frac=0.002, sigma_noise_I=0.5,
    temp_gradient=0.15, sigma_noise_T=0.2,
    capacity_drift_reset_soc=0.95,
    ocv_dispersion_floor=1e-3,
    # local estimator
    eta=0.995, q_used=Q_NOM_GROUP, current_deadband=2.0,
    knee_low=0.15, knee_high=0.85, rest_current=5.0, ocv_gain=0.01,
    ocv_slope_min=0.35, ocv_delta_max=0.02,
    v_full=3.55, pack_full_soc=0.95,
    quant_soc=0.001, quant_current=0.1,
)

# ---------------------------------------------------------------------------
# Two measured properties of this dataset drive the estimator's structure and
# are reported as findings (E10), not silently worked around:
#
# (1) The logged SOC channel is NOT consistent with an absolute LiFePO4 OCV
#     inversion of the logged cell voltages: corr(inv_OCV(V_pack/16), SOC_bms)
#     = -0.35, and where SOC_bms ~ 1 % the min/max cells sit at an OCV-implied
#     70 %/99 %.  Absolute voltage->SOC re-anchoring would drag every local
#     estimate by 40-70 SOC points, and a voltage-triggered full-charge reset
#     fires hundreds of times at low pack SOC.  Neither is usable here.
#
# (2) The logged current channel is not consistent with the logged SOC either:
#     integral(I dt) = +1600 Ah net over the window while SOC ends 9 points
#     BELOW where it started (mean I_pack = +3.2 A).  An open-loop Coulomb
#     counter therefore has ~0.55 SOC RMSE against the reference for ANY
#     assumed capacity (4480 / 9600 / 2000 Ah all give 0.55-0.60).
#
# Consequence: each node runs a DIFFERENTIAL estimator.  The common mode comes
# from the pack shunt (the Victron battery monitor SOC, which every node on the
# RS485 bus reads); what the node adds is its own module-level deviation,
# integrated from the difference between its own current measurement and the
# pack current, plus a bounded differential OCV correction that uses only the
# gap between its own cell voltage and the pack average cell voltage.  This is
# how a distributed BMS actually works, and it puts the experiment where the
# paper's claim lives: detecting NODE-LEVEL deviations, not absolute SOC.
# ---------------------------------------------------------------------------


# ------------------------------------------------------------ data loader
def load_pack_data(path=DATA_FILE, subsample=2):
    """Load the El Tiemblo CSV (two-row header) and subsample.

    dt is RECOMPUTED after subsampling from the timestamps so that
    integral(I dt) is physically correct; the round-1 code kept the dt of the
    full-rate frame and therefore under-counted charge throughput by ~2x.
    """
    df = pd.read_csv(path, header=[0, 1], low_memory=False)
    name_of = {
        'voltage': 'V_pack',
        'current': 'I_pack',
        'battery temperature': 'T_batt',
        'state of charge': 'SOC',
        'minimum cell voltage': 'V_cell_min',
        'maximum cell voltage': 'V_cell_max',
        'minimum cell temperature': 'T_cell_min',
        'maximum cell temperature': 'T_cell_max',
    }
    col = {}
    for i, (l1, l2) in enumerate(df.columns):
        k = str(l2).strip().lower()
        if 'Battery Monitor' in str(l1) and k in name_of:
            col[name_of[k]] = i
        if str(l2).strip() == 'Europe/Paris (+02:00)':
            col['timestamp'] = i

    out = pd.DataFrame()
    out['timestamp'] = pd.to_datetime(df.iloc[:, col['timestamp']])
    for k in ('V_pack', 'I_pack', 'T_batt', 'SOC',
              'V_cell_min', 'V_cell_max', 'T_cell_min', 'T_cell_max'):
        if k in col:
            out[k] = pd.to_numeric(df.iloc[:, col[k]], errors='coerce')
    out = out.dropna(subset=['V_pack', 'I_pack', 'SOC']).reset_index(drop=True)
    if subsample and subsample > 1:
        out = out.iloc[::subsample].reset_index(drop=True)
    dt = out['timestamp'].diff().dt.total_seconds()
    out['dt'] = dt.fillna(dt.median()).clip(0.1, 600.0)
    return out


def pack_arrays(df):
    """Extract the arrays the model needs, as plain numpy."""
    return dict(
        T=len(df),
        soc_bms=df['SOC'].values / 100.0,
        I=df['I_pack'].values.astype(float),
        V_pack=df['V_pack'].values.astype(float),
        T_batt=df['T_batt'].values.astype(float),
        V_min=df['V_cell_min'].values.astype(float),
        V_max=df['V_cell_max'].values.astype(float),
        dt=df['dt'].values.astype(float),
    )


# ---------------------------------------------------------------- model
class PackSignalModel:
    """Reconstruct 16 module signals + 16 honest local SOC estimates."""

    def __init__(self, data, seed, cfg=None, dispersion_scale=1.0,
                 anchor='ocv', physical_fault=None):
        """
        data              : dict from pack_arrays()
        seed              : int
        cfg               : dict overriding DEFAULTS
        dispersion_scale  : lambda, multiplies (sigma_Q, sigma_bias_I, sigma_e0)
        anchor            : 'ocv' | 'rank' | 'legacy'  (voltage reconstruction)
        physical_fault    : dict or None; see faults.py.  Keys used here:
                            'kind' in {aging, imbalance, short, thermal,
                                       i_offset, v_offset}
        """
        self.d = data
        self.cfg = dict(DEFAULTS)
        if cfg:
            self.cfg.update(cfg)
        self.rng = np.random.default_rng(seed)
        self.lam = float(dispersion_scale)
        self.anchor = anchor
        self.fault = physical_fault or {}
        self.n = N_MODULES
        self.frac_ocv_anchored = 0.0

    # -- parameter draw -------------------------------------------------
    def _draw(self):
        c, n, rng, lam = self.cfg, self.n, self.rng, self.lam
        p = {}
        p['delta_Q'] = rng.normal(0.0, c['sigma_Q'] * lam, n)
        p['rho_R0'] = rng.normal(0.0, c['sigma_R0'], n)
        p['e0'] = rng.normal(0.0, c['sigma_e0'] * lam, n)
        p['bias_V'] = rng.normal(0.0, c['sigma_bias_V'], n)
        p['bias_I'] = rng.normal(0.0, c['sigma_bias_I_frac'] * lam * c['I_FS'], n)
        f = self.fault
        tgt = f.get('target', None)
        if f.get('kind') == 'aging' and tgt is not None:
            p['delta_Q'][tgt] = -abs(f.get('capacity_loss', 0.15))
            p['rho_R0'][tgt] = f.get('r0_factor', 1.5) - 1.0
        if f.get('kind') == 'imbalance' and tgt is not None:
            p['e0'][tgt] += f.get('soc_offset', -0.08)
        p['Q'] = Q_NOM_GROUP * (1.0 + p['delta_Q'])
        p['R0'] = R0_GROUP * (1.0 + p['rho_R0'])
        return p

    # -- true SOC -------------------------------------------------------
    def _true_soc(self, p):
        d, c = self.d, self.cfg
        I, dt, soc_bms = d['I'], d['dt'], d['soc_bms']
        ah = np.cumsum(I * dt) / 3600.0                # Ah, + = charged in
        # reset the capacity-mismatch drift at every full-charge event
        full = soc_bms >= c['capacity_drift_reset_soc']
        ref = np.where(full, ah, np.nan)
        ref[0] = ref[0] if full[0] else 0.0
        ref = pd.Series(ref).ffill().fillna(0.0).values
        dah = (ah - ref)[:, None]                       # (T,1)
        drift = dah * (1.0 / p['Q'][None, :] - 1.0 / Q_NOM_GROUP)
        s = soc_bms[:, None] + p['e0'][None, :] + drift

        # internal short-circuit: leakage that does NOT cross the node's shunt
        leak_ah = np.zeros((len(ah), self.n))
        f = self.fault
        if f.get('kind') == 'short' and f.get('target') is not None:
            tgt, r_isc = f['target'], float(f.get('r_isc_ohm', 33.0))
            t0 = int(f.get('t_start', 0))
            act = np.zeros(len(ah)); act[t0:] = 1.0
            base = s[:, tgt].copy()
            s_it = np.clip(base, 0, 1)
            cum = np.zeros(len(ah))
            for _ in range(2):                          # 2 fixed-point passes
                i_leak = ocv_cell(s_it) / r_isc * act   # A, always discharging
                cum = np.cumsum(i_leak * dt) / 3600.0
                s_it = np.clip(base - cum / p['Q'][tgt], 0, 1)
            leak_ah[:, tgt] = cum
            s = s.copy()
            s[:, tgt] = base - cum / p['Q'][tgt]
        return np.clip(s, 0.0, 1.0), leak_ah

    # -- true voltage ---------------------------------------------------
    def _true_voltage(self, s_true, p):
        d, c = self.d, self.cfg
        V_min, V_max, I = d['V_min'], d['V_max'], d['I']
        env = (V_max - V_min)[:, None]
        n_ocv = 0
        if self.anchor == 'legacy':
            # round-1 scheme: fixed random position in the envelope,
            # independent of the module's physical state
            pos = np.linspace(0, 1, self.n)
            self.rng.shuffle(pos)
            u = np.repeat(pos[None, :], len(V_min), axis=0)
        else:
            order = np.argsort(np.argsort(s_true, axis=1), axis=1)
            u_rank = order / (self.n - 1.0)
            if self.anchor == 'rank':
                u = u_rank
            else:
                ocv = ocv_cell(s_true)                   # (T,n)
                lo = ocv.min(axis=1, keepdims=True)
                hi = ocv.max(axis=1, keepdims=True)
                spread = hi - lo
                u_ocv = np.divide(ocv - lo, np.where(spread > 0, spread, 1.0))
                use_ocv = (spread > c['ocv_dispersion_floor'])
                u = np.where(use_ocv, u_ocv, u_rank)
                n_ocv = int(use_ocv.sum())
        V = V_min[:, None] + u * env
        V = V + (p['R0'][None, :] - R0_GROUP) * I[:, None]
        self.frac_ocv_anchored = n_ocv / max(len(V_min), 1)
        return V

    # -- measured signals ----------------------------------------------
    def _measure(self, V_true, p):
        d, c, rng, n = self.d, self.cfg, self.rng, self.n
        T = d['T']
        V_hat = V_true + p['bias_V'][None, :] + rng.normal(0, c['sigma_noise_V'], (T, n))
        I_hat = (d['I'][:, None] + p['bias_I'][None, :]
                 + rng.normal(0, c['sigma_noise_I'], (T, n)))
        grad = (np.arange(n) - (n - 1) / 2.0) * c['temp_gradient']
        T_hat = d['T_batt'][:, None] + grad[None, :] + rng.normal(0, c['sigma_noise_T'], (T, n))

        f = self.fault
        tgt, t0 = f.get('target'), int(f.get('t_start', 0))
        if f.get('kind') == 'i_offset' and tgt is not None:
            I_hat[t0:, tgt] += float(f.get('amp_A', 5.0))
        if f.get('kind') == 'v_offset' and tgt is not None:
            V_hat[t0:, tgt] += float(f.get('amp_V', 0.050))
        if f.get('kind') == 'thermal' and tgt is not None:
            ramp = np.zeros(T)
            ramp[t0:] = np.linspace(0, float(f.get('dT_C', 5.0)), T - t0)
            T_hat[:, tgt] += ramp
        if f.get('kind') == 'short' and tgt is not None:
            T_hat[t0:, tgt] += float(f.get('dT_C', 0.5))
        # transducer quantisation of what actually goes on the bus
        I_hat = np.round(I_hat / c['quant_current']) * c['quant_current']
        return V_hat, I_hat, T_hat

    # -- honest local estimator ----------------------------------------
    def _local_estimator(self, V_hat, I_hat, s_true):
        """What each node's firmware computes and broadcasts.

        Differential Coulomb counting: the common mode is taken from the pack
        shunt (SOC_bms, read off the bus by every node) and the node adds its
        own module-level deviation, integrated from (own current - pack
        current) over the NOMINAL group capacity, with a BMS zero-current
        deadband.  A bounded differential OCV correction acts only at the OCV
        knees and only at rest, and the accumulated deviation is re-zeroed at
        pack full charge (what active balancing achieves).

        Returns the quantised report and a per-step event flag that the node
        also broadcasts in its status word, marking the steps where the
        counter was corrected or re-synchronised -- the aggregator must not
        read those as evidence of misbehaviour.
        """
        d, c, n = self.d, self.cfg, self.n
        T, dt = d['T'], d['dt']
        soc_bms, I_bus = d['soc_bms'], d['I']
        rep = np.zeros((T, n))
        ev = np.zeros((T, n), dtype=bool)
        s = np.clip(s_true[0].copy(), 0.0, 1.0)          # boot from real state
        rep[0] = np.round(s / c['quant_soc']) * c['quant_soc']
        s = rep[0].copy()
        ev[0] = True
        eta, q = c['eta'], c['q_used']
        db = c['current_deadband']
        n_ocv_corr = np.zeros(n, dtype=int)
        n_resync = np.zeros(n, dtype=int)
        for t in range(1, T):
            i_t = I_hat[t]
            i_eff = np.where(np.abs(i_t) < db, 0.0, i_t)      # BMS deadband
            i_bus = 0.0 if abs(I_bus[t]) < db else I_bus[t]
            ds_common = soc_bms[t] - soc_bms[t - 1]           # pack shunt
            ds_diff = eta * (i_eff - i_bus) * dt[t] / (3600.0 * q)
            s = s + ds_common + ds_diff
            e = np.zeros(n, dtype=bool)
            # bounded differential OCV correction: knees only, at rest only
            knee = (s < c['knee_low']) | (s > c['knee_high'])
            rest = np.abs(i_t) <= c['rest_current']
            m = knee & rest
            if m.any():
                v_corr = V_hat[t] - R0_GROUP * i_t
                dv = v_corr - np.median(v_corr)       # vs pack average cell
                slope = np.maximum(docv_dsoc(np.clip(s, 0, 1)), c['ocv_slope_min'])
                dsoc = np.clip(dv / slope, -c['ocv_delta_max'], c['ocv_delta_max'])
                # first-order pull toward "pack SOC + my voltage-implied offset"
                s_target = soc_bms[t] + dsoc
                s = np.where(m, s + c['ocv_gain'] * (s_target - s), s)
                e |= m
                n_ocv_corr += m
            # pack full-charge re-synchronisation (balancing zeroes the spread)
            if soc_bms[t] >= c['pack_full_soc'] and V_hat[t].max() >= c['v_full']:
                fc = V_hat[t] >= c['v_full']
                s = np.where(fc, soc_bms[t], s)
                e |= fc
                n_resync += fc
            clipped = (s < 0.0) | (s > 1.0)
            if clipped.any():
                e |= clipped
                s = np.clip(s, 0.0, 1.0)
            # the node keeps full internal precision and quantises only the
            # value it puts on the bus (JK BMS reports SOC in 0.1 % steps)
            rep[t] = np.round(s / c['quant_soc']) * c['quant_soc']
            ev[t] = e
        self.n_ocv_corr, self.n_resync = n_ocv_corr, n_resync
        return rep, ev

    # -- public ---------------------------------------------------------
    def build(self):
        p = self._draw()
        s_true, leak = self._true_soc(p)
        V_true = self._true_voltage(s_true, p)
        V_hat, I_hat, T_hat = self._measure(V_true, p)
        rep, ev = self._local_estimator(V_hat, I_hat, s_true)
        return dict(
            T=self.d['T'], dt=self.d['dt'], I_pack=self.d['I'],
            soc_bms=self.d['soc_bms'],
            s_true=s_true, soc_pack_true=s_true.mean(axis=1),
            V_true=V_true, V_hat=V_hat, I_hat=I_hat, T_hat=T_hat,
            s_hat=rep, event=ev, leak_ah=leak,
            params={k: p[k] for k in ('delta_Q', 'rho_R0', 'e0', 'bias_V',
                                      'bias_I', 'Q', 'R0')},
            frac_ocv_anchored=self.frac_ocv_anchored,
            n_ocv_corr=getattr(self, 'n_ocv_corr', None),
            n_resync=getattr(self, 'n_resync', None),
            lam=self.lam, anchor=self.anchor,
        )


def honest_dispersion(s_hat):
    """IQR and p95 of |s_m - median_m| over time, plus the cross-node MAD."""
    med = np.median(s_hat, axis=1, keepdims=True)
    dev = np.abs(s_hat - med)
    flat = dev.ravel()
    mad = np.median(dev, axis=1)
    return dict(
        iqr=float(np.percentile(flat, 75) - np.percentile(flat, 25)),
        p50=float(np.percentile(flat, 50)),
        p95=float(np.percentile(flat, 95)),
        p99=float(np.percentile(flat, 99)),
        mean=float(flat.mean()),
        max=float(flat.max()),
        mad_median=float(np.median(mad)),
        mad_p95=float(np.percentile(mad, 95)),
    )
