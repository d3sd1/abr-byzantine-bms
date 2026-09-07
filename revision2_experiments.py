#!/usr/bin/env python3
"""
revision2_experiments.py -- Applied Energy round 2 (APEN-D-26-09704R1 reject /
resubmit).  Rebuilds the experimental validation on top of the v3 signal model.

  E7   fault-class discrimination matrix (16 classes x 7 detectors)   [R1.1, R2]
  E8   honest-dispersion sweep, lambda in {0.25,0.5,1,2,4,8}          [R1.2]
  E9   fault magnitude vs honest spread                               [R1.2]
  E10  calibration against the real cell-voltage envelope             [R1.2]
  E11  voltage-anchor sensitivity -- replaces Table 11                [Table 11]
  E1'  detector ablation on the v3 model
  E2'  gap-threshold sweep on the v3 model
  E6'  centralised vs distributed on the v3 model

5 fixed seeds (42,123,456,789,1024), mean +/- std, every number straight from
the code.  Results that came out unfavourable are reported as they are.

Author: Andrei Garcia Cuadra (ETSIDI-UPM)
"""

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings('ignore')
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from signal_model import (  # noqa: E402
    load_pack_data, pack_arrays, PackSignalModel, honest_dispersion,
    ocv_cell, inv_ocv, N_MODULES, Q_NOM_GROUP, DATA_FILE)
from faults import build_scenario, CLASSES, ORDER, TARGET  # noqa: E402
from abr_detector import (  # noqa: E402
    ABRDetector, MedianMADDetector, FixedThresholdDetector, PBFTTrimmedMean,
    CentralisedMean, STATE_NAMES, TRUSTED, SUSPECT, QUARANTINED, BATTERY,
    message_freshness, router_stats)

MUST_NOT_QUARANTINE = [c for c in ORDER if CLASSES[c]['expected'] == 'no-quarantine']
NODE_CLASSES = [c for c in ORDER if CLASSES[c]['family'] == 'node']

SEEDS = [42, 123, 456, 789, 1024]
SUBSAMPLE = 2
OUT_DIR = SCRIPT_DIR / "results"
CONFIG = SCRIPT_DIR / "config_v3.json"
N = N_MODULES


# ----------------------------------------------------------------- helpers
def agg(vals):
    a = np.asarray([v for v in vals if v is not None], dtype=float)
    if a.size == 0:
        return {'mean': None, 'std': None, 'min': None, 'max': None, 'n': 0}
    return {'mean': float(a.mean()), 'std': float(a.std()),
            'min': float(a.min()), 'max': float(a.max()), 'n': int(a.size)}


def evaluate(res, sc):
    """Uniform metrics for every detector on one scenario."""
    tg = list(sc['targets'])
    ts = int(sc['t_start'])
    heal = [m for m in range(N) if m not in tg]
    excl, dt = res['excl'], sc['dt']
    cum_h = np.cumsum(dt) / 3600.0

    tpr = float(excl[ts:][:, tg].mean()) if tg else 0.0
    fpr = float(excl[:, heal].mean()) if heal else 0.0
    lat_s, lat_h = [], []
    for m in tg:
        f = int(res['first_ex'][m])
        if f >= 0:
            lat_s.append(float(max(f - ts, 0)))
            lat_h.append(float(max(cum_h[f] - cum_h[ts], 0.0)))
    rmse = float(np.sqrt(np.mean((res['cons'] - sc['soc_pack_true']) ** 2)))
    mae = float(np.mean(np.abs(res['cons'] - sc['soc_pack_true'])))

    dg = res.get('degraded')
    if dg is None:
        dg = np.zeros_like(excl)
    comms_t = float(dg[ts:][:, tg].mean()) if tg else 0.0
    comms_h = float(dg[:, heal].mean()) if heal else 0.0
    ever_c = res.get('ever_comms')
    if ever_c is None:
        ever_c = dg.any(axis=0)
    p_comms_t = float(np.mean([bool(ever_c[m]) for m in tg])) if tg else 0.0
    p_comms_h = float(np.mean([bool(ever_c[m]) for m in heal])) if heal else 0.0

    worst = res['worst']
    part = {k: 0.0 for k in STATE_NAMES}
    for m in tg:
        part[STATE_NAMES[int(worst[m])]] += 1.0 / max(len(tg), 1)
    heal_part = {k: 0.0 for k in STATE_NAMES}
    for m in heal:
        heal_part[STATE_NAMES[int(worst[m])]] += 1.0 / max(len(heal), 1)
    return dict(
        tpr_steps=tpr, fpr_steps=fpr,
        latency_steps=(float(np.mean(lat_s)) if lat_s else None),
        latency_hours=(float(np.mean(lat_h)) if lat_h else None),
        p_target_quarantined=part['quarantined'],
        p_target_battery=part['battery-anomaly'],
        p_target_suspect=part['suspect'],
        p_target_trusted=part['trusted'],
        p_healthy_quarantined=heal_part['quarantined'],
        p_healthy_battery=heal_part['battery-anomaly'],
        p_target_comms_degraded=p_comms_t,
        p_healthy_comms_degraded=p_comms_h,
        frac_steps_comms_target=comms_t,
        frac_steps_comms_healthy=comms_h,
        rmse_consensus=rmse, mae_consensus=mae)


def detector_bank(kw, full=True, persist=1):
    d = {
        'ABR-full': ABRDetector(router=True, cusum_persistence=persist, **kw),
        'ABR-round-1 (no router, no freshness)':
            ABRDetector(router=False, comms_state=False, **kw),
        'ABR-no-router': ABRDetector(router=False, **kw),
        'ABR-router-first': ABRDetector(router=True, router_first=True,
                                        cusum_persistence=persist, **kw),
        'Median/MAD z>3': MedianMADDetector(),
        'Fixed 3 %': FixedThresholdDetector(margin=0.03),
        'Fixed 5 %': FixedThresholdDetector(margin=0.05),
        'PBFT-BMS trimmed': PBFTTrimmedMean(n=N, f_max=5),
        'Centralised mean': CentralisedMean(n=N, f_max=5),
    }
    if not full:
        d = {k: d[k] for k in ('ABR-full', 'Median/MAD z>3',
                               'Fixed 3 %', 'Fixed 5 %')}
    return d


# ------------------------------------------------------- router calibration
def calibrate_router(data, cfg):
    """Thresholds from FAULT-FREE runs over the first 10 % of the horizon.

    Never touched afterwards, and never informed by any fault class.
    """
    rc = cfg['detector']['router']
    frac = rc['calibration_window_fraction']
    pct = rc['calibration_percentile']
    margin = rc['calibration_margin']
    eta = rc['cusum_eta']
    q = rc['cusum_q_Ah']
    db = cfg['local_estimator']['current_deadband_A']
    a = rc['current_consistency_ewma_alpha']
    floor = rc['current_mad_floor_A']

    r_pool, ci_pool = [], []
    for seed in SEEDS:
        sc = build_scenario(data, None, seed)
        T = sc['T']
        W = int(frac * T)
        s, i, ev = sc['s_rep'], sc['i_rep'], sc['ev_rx']
        dt, soc, ib = sc['dt'], sc['soc_bms'], sc['I_pack']
        i_eff = np.where(np.abs(i) < db, 0.0, i)
        ibe = np.where(np.abs(ib) < db, 0.0, ib)
        ds = np.diff(soc, prepend=soc[0])[:, None]
        exp_d = ds + eta * (i_eff - ibe[:, None]) * dt[:, None] / (3600.0 * q)
        r = np.where(ev, 0.0, np.diff(s, axis=0, prepend=s[:1]) - exp_d)
        r[0] = 0.0
        r_pool.append(np.abs(r[:W]).ravel())
        mi = np.median(i, axis=1, keepdims=True)
        mad = np.maximum(np.median(np.abs(i - mi), axis=1, keepdims=True) * 1.4826, floor)
        ci = np.abs(i - mi) / mad
        acc = np.zeros(N)
        h = np.zeros((W, N))
        for t in range(W):
            acc = (1 - a) * acc + a * ci[t]
            h[t] = acc
        ci_pool.append(h.ravel())

    r_pool = np.concatenate(r_pool)
    ci_pool = np.concatenate(ci_pool)
    slack = float(np.percentile(r_pool, pct))
    th_ci = float(margin * np.percentile(ci_pool, pct))

    # honest CUSUM under that slack, same window
    mx = []
    for seed in SEEDS:
        sc = build_scenario(data, None, seed)
        T = sc['T']
        W = int(frac * T)
        s, i, ev = sc['s_rep'], sc['i_rep'], sc['ev_rx']
        dt, soc, ib = sc['dt'], sc['soc_bms'], sc['I_pack']
        i_eff = np.where(np.abs(i) < db, 0.0, i)
        ibe = np.where(np.abs(ib) < db, 0.0, ib)
        ds = np.diff(soc, prepend=soc[0])[:, None]
        exp_d = ds + eta * (i_eff - ibe[:, None]) * dt[:, None] / (3600.0 * q)
        r = np.where(ev, 0.0, np.diff(s, axis=0, prepend=s[:1]) - exp_d)
        r[0] = 0.0
        Cp = np.cumsum(r[:W] - slack, axis=0)
        Sp = Cp - np.minimum.accumulate(np.minimum(Cp, 0.0), axis=0)
        Cn = np.cumsum(-r[:W] - slack, axis=0)
        Sn = Cn - np.minimum.accumulate(np.minimum(Cn, 0.0), axis=0)
        mx.append(float(np.maximum(Sp, Sn).max()))
    th_cusum = float(margin * max(max(mx), 1e-9))

    # longest honest excursion above the CUSUM threshold, fault-free, both in
    # the calibration window and over the whole horizon
    longest_win = longest_full = 0
    for seed in SEEDS:
        sc = build_scenario(data, None, seed)
        T = sc['T']
        W = int(frac * T)
        deg = message_freshness(sc, cfg['detector']['router'].get(
            'comms_max_age_steps', 2))
        _, cus = router_stats(sc, ci_alpha=a, ci_mad_floor=floor,
                              cusum_slack=slack, eta=eta, q_nom=q,
                              deadband=db, degraded=deg)
        raw = cus > th_cusum
        acc = np.zeros(N, dtype=int)
        for t in range(T):
            acc = np.where(raw[t], acc + 1, 0)
            m = int(acc.max())
            if t < W:
                longest_win = max(longest_win, m)
            longest_full = max(longest_full, m)
    p_cal = max(1, int(np.ceil(margin * longest_win)))

    out = dict(threshold_current_consistency=th_ci, threshold_cusum=th_cusum,
               cusum_slack=slack,
               persistence_from_calibration=p_cal,
               honest_longest_excursion_window=int(longest_win),
               honest_longest_excursion_full_horizon=int(longest_full),
               honest_pooled_abs_residual_p999=float(np.percentile(r_pool, pct)),
               honest_pooled_ewma_ci_p999=float(np.percentile(ci_pool, pct)),
               honest_max_cusum_in_window=float(max(mx)),
               honest_max_ewma_ci_in_window=float(ci_pool.max()))
    print(f"  router calibration: th_ci={th_ci:.3f}  th_cusum={th_cusum:.3e}  "
          f"slack={slack:.3e}")
    print(f"  honest CUSUM excursions above threshold: longest {longest_win} steps "
          f"in the calibration window, {longest_full} over the full horizon "
          f"-> P from calibration = {p_cal}")
    return out


def detector_kwargs(cal, cfg):
    dc = cfg['detector']
    return dict(
        n=N, f_max=dc['f_max'], mad_floor=dc['mad_floor'],
        alpha_up=dc['ewma_alpha_up'], alpha_down=dc['ewma_alpha_down'],
        ewma_cap=dc['ewma_cap'], sigmoid_k=dc['trust_sigmoid_k'],
        sigmoid_x0=dc['trust_sigmoid_x0'], trust_high=dc['trust_high'],
        trust_low=dc['trust_low'], trust_battery=dc['trust_battery_anomaly'],
        gap_threshold=dc['gap_threshold'],
        min_ewma_quarantine=dc['min_ewma_quarantine'],
        suspect_z=dc['suspect_z'],
        ci_alpha=dc['router']['current_consistency_ewma_alpha'],
        ci_mad_floor=dc['router']['current_mad_floor_A'],
        th_ci=cal['threshold_current_consistency'],
        th_cusum=cal['threshold_cusum'], cusum_slack=cal['cusum_slack'],
        eta=dc['router']['cusum_eta'], q_nom=dc['router']['cusum_q_Ah'],
        deadband=cfg['local_estimator']['current_deadband_A'])


# ======================================================================
# E12 -- CUSUM persistence sweep and configuration selection (iteration 1)
# ======================================================================
# Selection criterion, declared BEFORE looking at any outcome:
#   (hard)  false-positive rate exactly 0 on the fault-free control AND
#           zero quarantines on the eight classes that must not be quarantined
#   (score) among the configurations that pass, the highest mean probability of
#           quarantining the target across the eight node-fault classes.
# If none passes, the iteration-0 configuration stands.
P_GRID = [1, 2, 30, 520]


def _passes(res):
    if res['none']['fpr_steps']['mean'] > 0:
        return False
    if res['none']['p_target_quarantined']['mean'] > 0:
        return False
    for c in MUST_NOT_QUARANTINE:
        if res[c]['p_target_quarantined']['mean'] > 0:
            return False
        if res[c]['fpr_steps']['mean'] > 0:
            return False
    return True


def _score(res):
    return float(np.mean([res[c]['p_target_quarantined']['mean']
                          for c in NODE_CLASSES]))


def exp12_persistence(data, kw):
    print()
    print("=" * 72)
    print("E12 CUSUM persistence sweep + configuration selection (iteration 1)")
    print("=" * 72)
    variants = ([('router-first', True, P) for P in P_GRID]
                + [('gap-first (v3)', False, P) for P in (1, 520)])
    out = {'grid': P_GRID, 'criterion':
           'FPR==0 on the control and zero quarantines on the eight '
           'must-not-quarantine classes; then max mean P(quarantine) over the '
           'eight node-fault classes', 'variants': {}}
    for label, rfirst, P in variants:
        key = f'{label} P={P}'
        per = {}
        for name in ORDER + ['none']:
            cls = None if name == 'none' else name
            lst = []
            for seed in SEEDS:
                sc = build_scenario(data, cls, seed)
                if cls is None:
                    sc['targets'] = [TARGET]
                det = ABRDetector(router=True, router_first=rfirst,
                                  cusum_persistence=P, **kw)
                lst.append(evaluate(det.run(sc), sc))
            per[name] = {k: agg([d[k] for d in lst]) for k in lst[0]}
        ok = _passes(per)
        sc_ = _score(per)
        out['variants'][key] = {'per_class': per, 'passes_criterion': bool(ok),
                                'node_fault_score': sc_,
                                'router_first': rfirst, 'persistence': P}
        wrong = float(np.mean([per[c]['p_target_quarantined']['mean']
                               for c in MUST_NOT_QUARANTINE]))
        print(f"  {key:24s} passes={str(ok):5s} node-fault P(quar)={sc_:.3f} "
              f"wrong-quarantine={wrong:.3f} "
              f"ctrlFPR={per['none']['fpr_steps']['mean']:.4f} "
              f"| N6={per['N6_collusion']['p_target_quarantined']['mean']:.2f} "
              f"G3={per['G3_spikes']['p_target_quarantined']['mean']:.2f} "
              f"N5={per['N5_firmware_bias']['p_target_quarantined']['mean']:.2f}")
    passing = {k: v for k, v in out['variants'].items() if v['passes_criterion']}
    if passing:
        best = max(passing, key=lambda k: passing[k]['node_fault_score'])
    else:
        best = 'gap-first (v3) P=1'
        print("  no configuration passes the hard constraint; "
              "the iteration-0 configuration stands")
    out['selected'] = best
    out['selected_router_first'] = out['variants'][best]['router_first']
    out['selected_persistence'] = out['variants'][best]['persistence']
    print(f"  SELECTED: {best}")
    return out


# ======================================================================
# E7 -- fault-class discrimination matrix
# ======================================================================
def exp7(data, kw, persist=1, router_first=False):
    print("\n" + "=" * 72)
    print("E7  Fault-class discrimination matrix (R1.1 / R2)")
    print("=" * 72)
    out = {}
    for name in ORDER + ['none']:
        cls = None if name == 'none' else name
        per = {}
        for seed in SEEDS:
            sc = build_scenario(data, cls, seed)
            if cls is None:
                sc['targets'] = [TARGET]        # score the same node
            bank = detector_bank(kw, persist=persist)
            if router_first:
                bank['ABR-full'] = ABRDetector(router=True, router_first=True,
                                               cusum_persistence=persist, **kw)
            for dname, det in bank.items():
                m = evaluate(det.run(sc), sc)
                per.setdefault(dname, []).append(m)
        out[name] = {}
        for dname, lst in per.items():
            keys = lst[0].keys()
            out[name][dname] = {k: agg([d[k] for d in lst]) for k in keys}
            out[name][dname]['per_seed'] = {
                k: [d[k] for d in lst] for k in
                ('tpr_steps', 'fpr_steps', 'rmse_consensus',
                 'p_target_quarantined', 'p_target_battery')}
        spec = CLASSES.get(name, {})
        a = out[name]['ABR-full']
        print(f"  {name:28s} exp={spec.get('expected','none'):14s} "
              f"ABR: Q={a['p_target_quarantined']['mean']:.2f} "
              f"B={a['p_target_battery']['mean']:.2f} "
              f"S={a['p_target_suspect']['mean']:.2f} "
              f"T={a['p_target_trusted']['mean']:.2f} "
              f"| noRt Q={out[name]['ABR-no-router']['p_target_quarantined']['mean']:.2f} "
              f"| rF Q={out[name]['ABR-router-first']['p_target_quarantined']['mean']:.2f} "
              f"| fix3 tpr={out[name]['Fixed 3 %']['tpr_steps']['mean']:.2f} "
              f"| RMSE={a['rmse_consensus']['mean']:.4f}")
    return out


# ======================================================================
# E8 -- honest-dispersion sweep
# ======================================================================
def exp8(data, kw, persist=1, router_first=False):
    print("\n" + "=" * 72)
    print("E8  Honest-dispersion sweep (R1.2)")
    print("=" * 72)
    lams = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
    classes = ['N5_firmware_bias', 'N1_current_offset', 'B1_aging', 'G1_packet_loss']
    dets = ['ABR-full', 'Fixed 3 %', 'Fixed 5 %', 'Median/MAD z>3']
    out = {'lambdas': lams, 'dispersion': {}, 'results': {}}
    for lam in lams:
        disp = []
        for seed in SEEDS:
            b = PackSignalModel(data, seed, dispersion_scale=lam).build()
            disp.append(honest_dispersion(b['s_hat']))
        out['dispersion'][str(lam)] = {k: agg([d[k] for d in disp])
                                       for k in disp[0]}
        print(f"  lambda={lam:<5} honest IQR={out['dispersion'][str(lam)]['iqr']['mean']:.4f} "
              f"p95={out['dispersion'][str(lam)]['p95']['mean']:.4f} "
              f"MAD={out['dispersion'][str(lam)]['mad_median']['mean']:.4f}")
        for cls in classes:
            per = {}
            for seed in SEEDS:
                sc = build_scenario(data, cls, seed, lam=lam)
                bank = detector_bank(kw, full=False, persist=persist)
                if router_first:
                    bank['ABR-full'] = ABRDetector(router=True, router_first=True,
                                                   cusum_persistence=persist, **kw)
                for dname in dets:
                    per.setdefault(dname, []).append(evaluate(bank[dname].run(sc), sc))
            for dname, lst in per.items():
                out['results'].setdefault(cls, {}).setdefault(dname, {})[str(lam)] = {
                    k: agg([d[k] for d in lst]) for k in lst[0]}
            r = out['results'][cls]
            print(f"     {cls:22s} " + "  ".join(
                f"{d.split()[0][:6]}:tpr={r[d][str(lam)]['tpr_steps']['mean']:.2f}"
                f"/fpr={r[d][str(lam)]['fpr_steps']['mean']:.3f}" for d in dets))
    return out


# ======================================================================
# E9 -- fault magnitude vs honest spread
# ======================================================================
def exp9(data, kw, persist=1, router_first=False):
    print("\n" + "=" * 72)
    print("E9  Fault magnitude vs honest spread (R1.2)")
    print("=" * 72)
    out = {'N5_bias': {}, 'N1_current': {}, 'honest_mad': None}
    mads = []
    for seed in SEEDS:
        b = PackSignalModel(data, seed, dispersion_scale=1.0).build()
        mads.append(honest_dispersion(b['s_hat'])['mad_median'])
    out['honest_mad'] = agg(mads)
    mad_mean = out['honest_mad']['mean']
    print(f"  honest cross-node MAD at lambda=1: {mad_mean:.4f} SOC")

    for bias in [0.01, 0.02, 0.03, 0.05, 0.10]:
        per = {}
        for seed in SEEDS:
            sc = build_scenario(data, 'N5_firmware_bias', seed,
                                msg_override={'bias': bias})
            bank = detector_bank(kw, full=False, persist=persist)
            if router_first:
                bank['ABR-full'] = ABRDetector(router=True, router_first=True,
                                               cusum_persistence=persist, **kw)
            for dname in ('ABR-full', 'Fixed 3 %', 'Median/MAD z>3'):
                per.setdefault(dname, []).append(evaluate(bank[dname].run(sc), sc))
        out['N5_bias'][str(bias)] = {
            d: {k: agg([x[k] for x in lst]) for k in lst[0]} for d, lst in per.items()}
        out['N5_bias'][str(bias)]['bias_in_mad_units'] = float(bias / max(mad_mean, 1e-9))
        a = out['N5_bias'][str(bias)]['ABR-full']
        print(f"    N5 bias={bias*100:5.1f} % ({bias/mad_mean:5.2f} MAD): "
              f"ABR tpr={a['tpr_steps']['mean']:.3f} Q={a['p_target_quarantined']['mean']:.2f} "
              f"lat={a['latency_steps']['mean'] if a['latency_steps']['mean'] else -1:.0f} steps "
              f"| fix3 tpr={out['N5_bias'][str(bias)]['Fixed 3 %']['tpr_steps']['mean']:.3f}")
    for amp in [2.0, 5.0, 10.0]:
        per = {}
        for seed in SEEDS:
            sc = build_scenario(data, 'N1_current_offset', seed,
                                sig_override={'amp_A': amp})
            bank = detector_bank(kw, full=False, persist=persist)
            if router_first:
                bank['ABR-full'] = ABRDetector(router=True, router_first=True,
                                               cusum_persistence=persist, **kw)
            for dname in ('ABR-full', 'Fixed 3 %', 'Median/MAD z>3'):
                per.setdefault(dname, []).append(evaluate(bank[dname].run(sc), sc))
        out['N1_current'][str(amp)] = {
            d: {k: agg([x[k] for x in lst]) for k in lst[0]} for d, lst in per.items()}
        a = out['N1_current'][str(amp)]['ABR-full']
        print(f"    N1 offset={amp:5.1f} A: ABR tpr={a['tpr_steps']['mean']:.3f} "
              f"Q={a['p_target_quarantined']['mean']:.2f} "
              f"lat={a['latency_steps']['mean'] if a['latency_steps']['mean'] else -1:.0f} steps")
    return out


# ======================================================================
# E10 -- calibration against the real envelope
# ======================================================================
def exp10(data, df):
    print("\n" + "=" * 72)
    print("E10 Calibration against the measured cell-voltage envelope (R1.2)")
    print("=" * 72)
    V_min, V_max = data['V_min'], data['V_max']
    env = V_max - V_min
    I, soc, dt = data['I'], data['soc_bms'], data['dt']
    rest = np.abs(I) <= 5.0
    meas = dict(
        envelope_mean_V=float(env.mean()), envelope_median_V=float(np.median(env)),
        envelope_p95_V=float(np.percentile(env, 95)), envelope_max_V=float(env.max()),
        envelope_rest_mean_V=float(env[rest].mean()),
        envelope_rest_p95_V=float(np.percentile(env[rest], 95)))
    # dataset self-consistency checks
    ds = np.diff(soc)
    up, dn = ds > 0, ds < 0
    cons = dict(
        corr_invOCV_Vpack_vs_SOC=float(np.corrcoef(inv_ocv(data['V_pack'] / 16), soc)[0, 1]),
        mean_I_when_SOC_rises_A=float(I[1:][up].mean()),
        mean_I_when_SOC_falls_A=float(I[1:][dn].mean()),
        net_charge_throughput_Ah=float((I * dt).sum() / 3600.0),
        net_SOC_change=float(soc[-1] - soc[0]),
        invOCV_Vmin_where_SOC_below_015=float(inv_ocv(V_min[soc < 0.15]).mean()),
        invOCV_Vmax_where_SOC_below_015=float(inv_ocv(V_max[soc < 0.15]).mean()),
        SOC_mean_where_SOC_below_015=float(soc[soc < 0.15].mean()),
        horizon_hours=float(dt.sum() / 3600.0), n_samples=int(len(soc)))
    # --- iteration 1: segment-wise capacity consistency -------------------
    # Comparing NET charge throughput with the NET SOC change over 70 days is
    # not a valid consistency test, because the BMS re-anchors its counter at
    # every full charge.  The test is redone between re-anchoring events: the
    # series is cut wherever SOC_bms jumps (|dSOC| > 2 points in one poll, or
    # SOC_bms >= 95 %) and wherever the log has a gap, and within each segment
    # dSOC_bms is regressed on the charge integral.
    jump = np.abs(np.diff(soc, prepend=soc[0])) > 0.02
    full = soc >= 0.95
    gap = dt >= 300.0
    cut = jump | full | gap
    seg_id = np.cumsum(cut)
    ah = np.cumsum(I * dt) / 3600.0
    segs = []
    for sid in np.unique(seg_id):
        m = seg_id == sid
        idx = np.flatnonzero(m)
        if idx.size < 20:
            continue
        d_soc = soc[idx[-1]] - soc[idx[0]]
        d_ah = ah[idx[-1]] - ah[idx[0]]
        if abs(d_ah) < 5.0:
            continue
        segs.append((d_soc, d_ah, idx.size))
    seg_fit = {'n_segments_usable': len(segs)}
    if segs:
        y = np.array([a for a, _, _ in segs])
        for q_name, q_val in (('280Ah_cell', 280.0), ('4480Ah_group', 4480.0)):
            x = np.array([b for _, b, _ in segs]) / q_val
            beta = float(np.sum(x * y) / np.sum(x * x))
            resid = y - beta * x
            ss_tot = float(np.sum((y - y.mean()) ** 2))
            r2 = float(1.0 - np.sum(resid ** 2) / ss_tot) if ss_tot > 0 else float('nan')
            seg_fit[q_name] = {'slope_beta': beta,
                               'implied_effective_capacity_Ah': float(q_val / beta)
                               if beta != 0 else None,
                               'r2': r2}
        seg_fit['pearson_r_dSOC_vs_dAh'] = float(
            np.corrcoef(y, np.array([b for _, b, _ in segs]))[0, 1])
        seg_fit['total_segment_samples'] = int(sum(c for _, _, c in segs))

    # --- iteration 1: how inconsistent is the voltage channel, and where ----
    v_mid = 0.5 * (V_min + V_max)
    err = np.abs(inv_ocv(v_mid) - soc)
    rest_m = np.abs(I) <= 5.0
    t_h = np.cumsum(dt) / 3600.0
    day = np.floor(t_h / 24.0).astype(int)
    bad = err > 0.20
    per_day = {}
    for dd in np.unique(day):
        m = day == dd
        per_day[int(dd)] = float(bad[m].mean())
    days_sorted = sorted(per_day)
    half = len(days_sorted) // 2
    ocv_incons = {
        'frac_samples_err_gt_20pts_all': float(bad.mean()),
        'frac_samples_err_gt_20pts_at_rest': float(bad[rest_m].mean()),
        'median_abs_err_all': float(np.median(err)),
        'median_abs_err_at_rest': float(np.median(err[rest_m])),
        'frac_in_first_half_of_coverage': float(
            np.mean([per_day[d] for d in days_sorted[:half]])) if half else None,
        'frac_in_second_half_of_coverage': float(
            np.mean([per_day[d] for d in days_sorted[half:]])) if half else None,
        'per_day_fraction': per_day,
        'n_days_of_coverage': len(days_sorted),
    }

    sim = {}
    for lam in [0.25, 1.0, 4.0]:
        dv, ds_, tv = [], [], []
        for seed in SEEDS:
            b = PackSignalModel(data, seed, dispersion_scale=lam).build()
            dv.append(float((b['V_hat'].max(1) - b['V_hat'].min(1)).mean()))
            tv.append(float(np.percentile(b['V_hat'].max(1) - b['V_hat'].min(1), 95)))
            ds_.append(honest_dispersion(b['s_hat']))
        sim[str(lam)] = dict(
            sim_voltage_spread_mean_V=agg(dv), sim_voltage_spread_p95_V=agg(tv),
            soc_dispersion={k: agg([d[k] for d in ds_]) for k in ds_[0]})
    print(f"  measured envelope: mean {meas['envelope_mean_V']:.3f} V, "
          f"p95 {meas['envelope_p95_V']:.3f} V")
    print(f"  simulated voltage spread (lambda=1): mean "
          f"{sim['1.0']['sim_voltage_spread_mean_V']['mean']:.3f} V, "
          f"p95 {sim['1.0']['sim_voltage_spread_p95_V']['mean']:.3f} V")
    print(f"  simulated SOC dispersion (lambda=1): IQR "
          f"{sim['1.0']['soc_dispersion']['iqr']['mean']:.4f}, p95 "
          f"{sim['1.0']['soc_dispersion']['p95']['mean']:.4f}")
    print(f"  dataset self-consistency: corr(invOCV(Vpack/16), SOC) = "
          f"{cons['corr_invOCV_Vpack_vs_SOC']:+.3f}; net throughput "
          f"{cons['net_charge_throughput_Ah']:.0f} Ah vs net SOC change "
          f"{cons['net_SOC_change']:+.3f}")
    print(f"  segment-wise capacity fit ({seg_fit['n_segments_usable']} segments): "
          + (", ".join(f"Q={k}: beta={v['slope_beta']:.3f} R2={v['r2']:.3f} "
                       f"Qeff={v['implied_effective_capacity_Ah']:.0f} Ah"
                       for k, v in seg_fit.items()
                       if isinstance(v, dict) and 'slope_beta' in v)
             if seg_fit['n_segments_usable'] else "no usable segment"))
    print(f"  OCV inconsistency: |invOCV(V_mid)-SOC| > 20 pts in "
          f"{ocv_incons['frac_samples_err_gt_20pts_all']*100:.1f} % of samples "
          f"({ocv_incons['frac_samples_err_gt_20pts_at_rest']*100:.1f} % at rest); "
          f"first half of coverage {ocv_incons['frac_in_first_half_of_coverage']*100:.1f} % "
          f"vs second half {ocv_incons['frac_in_second_half_of_coverage']*100:.1f} %")
    return dict(measured_envelope=meas, dataset_consistency=cons, simulated=sim,
                segmentwise_capacity_fit=seg_fit,
                ocv_inconsistency=ocv_incons)


# ======================================================================
# E11 -- voltage-anchor sensitivity (replaces Table 11)
# ======================================================================
def exp11(data, kw, persist=1, router_first=False):
    print("\n" + "=" * 72)
    print("E11 Voltage-anchor sensitivity -- replaces Table 11")
    print("=" * 72)
    out = {}
    for anchor in ['ocv', 'rank', 'legacy']:
        out[anchor] = {}
        for cls in [None, 'N5_firmware_bias', 'B1_aging']:
            per, disp = [], []
            for seed in SEEDS:
                sc = build_scenario(data, cls, seed, anchor=anchor)
                if cls is None:
                    sc['targets'] = [TARGET]
                per.append(evaluate(ABRDetector(
                    router=True, router_first=router_first,
                    cusum_persistence=persist, **kw).run(sc), sc))
                disp.append(honest_dispersion(sc['s_hat']))
            key = 'none' if cls is None else cls
            out[anchor][key] = {k: agg([d[k] for d in per]) for k in per[0]}
            out[anchor][key]['honest_dispersion'] = {
                k: agg([d[k] for d in disp]) for k in disp[0]}
            r = out[anchor][key]
            print(f"  anchor={anchor:7s} {key:20s} RMSE={r['rmse_consensus']['mean']:.5f} "
                  f"Q={r['p_target_quarantined']['mean']:.2f} "
                  f"B={r['p_target_battery']['mean']:.2f} "
                  f"IQR={r['honest_dispersion']['iqr']['mean']:.4f}")
    return out


# ======================================================================
# E1' / E2' / E6'
# ======================================================================
ABL_CLASSES = ['N5_firmware_bias', 'N8_random_report',
               'N7_stealth_drift', 'N6_collusion']


def exp1p(data, kw, persist=1, router_first=False):
    print("\n" + "=" * 72)
    print("E1' Detector ablation on the v3 signal model")
    print("=" * 72)
    variants = {
        'EWMA trust only': dict(router=False, gap_quarantine=False, trust_grading=True),
        'Gap only (no trust grading)': dict(router=False, gap_quarantine=True, trust_grading=False),
        'EWMA + gap (round-1 detector)': dict(router=False, gap_quarantine=True,
                                              trust_grading=True, comms_state=False),
        'EWMA + gap + message freshness': dict(router=False, gap_quarantine=True,
                                               trust_grading=True),
        'EWMA + gap + router (v3)': dict(router=True, gap_quarantine=True,
                                         trust_grading=True,
                                         router_first=router_first,
                                         cusum_persistence=persist),
    }
    out = {}
    for cls in ABL_CLASSES:
        out[cls] = {}
        per = {v: [] for v in variants}
        for seed in SEEDS:
            sc = build_scenario(data, cls, seed)
            for v, opt in variants.items():
                per[v].append(evaluate(ABRDetector(**opt, **kw).run(sc), sc))
        for v, lst in per.items():
            out[cls][v] = {k: agg([d[k] for d in lst]) for k in lst[0]}
            r = out[cls][v]
            print(f"  {cls:22s} {v:30s} RMSE={r['rmse_consensus']['mean']:.4f} "
                  f"tpr={r['tpr_steps']['mean']:.3f} fpr={r['fpr_steps']['mean']:.4f}")
    # the same ablation on the classes that must NOT be quarantined
    out['_false_positive_check'] = {}
    for cls in ['B1_aging', 'B2_imbalance', 'G1_packet_loss', 'G2_delay']:
        out['_false_positive_check'][cls] = {}
        per = {v: [] for v in variants}
        for seed in SEEDS:
            sc = build_scenario(data, cls, seed)
            for v, opt in variants.items():
                per[v].append(evaluate(ABRDetector(**opt, **kw).run(sc), sc))
        for v, lst in per.items():
            out['_false_positive_check'][cls][v] = {
                k: agg([d[k] for d in lst]) for k in lst[0]}
        print(f"  [must NOT quarantine] {cls:20s} " + "  ".join(
            f"{v.split()[0]}:{out['_false_positive_check'][cls][v]['p_target_quarantined']['mean']:.2f}"
            for v in variants))
    return out


def exp2p(data, kw, persist=1, router_first=False):
    print("\n" + "=" * 72)
    print("E2' Gap-threshold sweep on the v3 signal model")
    print("=" * 72)
    gaps = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
    classes = ABL_CLASSES + ['B1_aging', 'B2_imbalance', 'G1_packet_loss']
    acc = {g: {'tpr': [], 'fpr': [], 'q_node': [], 'q_batt': [], 'rmse': []}
           for g in gaps}
    for cls in classes:
        for seed in SEEDS:
            sc = build_scenario(data, cls, seed)
            node_fault = CLASSES[cls]['family'] == 'node'
            for g in gaps:
                k2 = dict(kw)
                k2['gap_threshold'] = g
                m = evaluate(ABRDetector(
                    router=True, router_first=router_first,
                    cusum_persistence=persist, **k2).run(sc), sc)
                acc[g]['rmse'].append(m['rmse_consensus'])
                acc[g]['fpr'].append(m['fpr_steps'])
                if node_fault:
                    acc[g]['tpr'].append(m['tpr_steps'])
                    acc[g]['q_node'].append(m['p_target_quarantined'])
                else:
                    acc[g]['q_batt'].append(m['p_target_quarantined'])
    out = {'gaps': gaps, 'sweep': {}}
    for g in gaps:
        out['sweep'][str(g)] = {k: agg(v) for k, v in acc[g].items()}
        s = out['sweep'][str(g)]
        print(f"  gap>{g:<4}: TPR={s['tpr']['mean']:.3f} FPR={s['fpr']['mean']:.4f} "
              f"P(quar|node fault)={s['q_node']['mean']:.2f} "
              f"P(quar|NOT a node fault)={s['q_batt']['mean']:.2f} "
              f"RMSE={s['rmse']['mean']:.4f}")
    return out


def exp6p(data, kw, persist=1, router_first=False):
    print("\n" + "=" * 72)
    print("E6' Centralised vs distributed on the v3 signal model")
    print("=" * 72)
    out = {}
    for cls in ABL_CLASSES:
        cen, dis, cenm, dism = [], [], [], []
        for seed in SEEDS:
            sc = build_scenario(data, cls, seed)
            c = evaluate(CentralisedMean(n=N, f_max=5).run(sc), sc)
            d = evaluate(ABRDetector(
                router=True, router_first=router_first,
                cusum_persistence=persist, **kw).run(sc), sc)
            cen.append(c['rmse_consensus']); dis.append(d['rmse_consensus'])
            cenm.append(c['mae_consensus']); dism.append(d['mae_consensus'])
        out[cls] = {'Centralised mean': {'rmse': agg(cen), 'mae': agg(cenm)},
                    'Distributed ABR': {'rmse': agg(dis), 'mae': agg(dism)}}
        c, d = np.mean(cen), np.mean(dis)
        out[cls]['degradation_factor'] = float(c / max(d, 1e-12))
        print(f"  {cls:22s} centralised RMSE={c:.4f}  distributed RMSE={d:.4f}  "
              f"(x{c/max(d,1e-12):.1f})")
    return out


# ======================================================================
def main():
    t0 = time.time()
    cfg = json.loads(CONFIG.read_text(encoding='utf-8'))
    print("=" * 72)
    print("ABR distributed BMS -- Applied Energy round 2 experiments (v3 model)")
    print(f"Seeds: {SEEDS} | subsample={SUBSAMPLE}")
    print("=" * 72)
    df = load_pack_data(DATA_FILE, subsample=SUBSAMPLE)
    data = pack_arrays(df)
    print(f"Loaded {len(df)} samples, horizon {data['dt'].sum()/3600:.1f} h "
          f"({data['dt'].sum()/86400:.1f} days of covered time)")

    print("\nCalibrating the router on fault-free data (first 10 % of horizon)...")
    cal = calibrate_router(data, cfg)
    cfg['detector']['router'].update(
        threshold_current_consistency=cal['threshold_current_consistency'],
        threshold_cusum=cal['threshold_cusum'],
        cusum_slack=cal['cusum_slack'],
        persistence_from_calibration=cal['persistence_from_calibration'],
        honest_longest_excursion_window=cal['honest_longest_excursion_window'],
        honest_longest_excursion_full_horizon=cal[
            'honest_longest_excursion_full_horizon'])
    kw = detector_kwargs(cal, cfg)

    R = {'_meta': {}, 'router_calibration': cal}
    R['E12_persistence_selection'] = exp12_persistence(data, kw)
    P = R['E12_persistence_selection']['selected_persistence']
    RF = R['E12_persistence_selection']['selected_router_first']
    print()
    print(f"Running the rest with the selected configuration: "
          f"router_first={RF}, persistence P={P}")
    R['E7_fault_class_matrix'] = exp7(data, kw, P, RF)
    R['E8_dispersion_sweep'] = exp8(data, kw, P, RF)
    R['E9_magnitude_vs_spread'] = exp9(data, kw, P, RF)
    R['E10_calibration'] = exp10(data, df)
    R['E11_voltage_anchor'] = exp11(data, kw, P, RF)
    R['E1p_ablation'] = exp1p(data, kw, P, RF)
    R['E2p_gap_sweep'] = exp2p(data, kw, P, RF)
    R['E6p_centralised'] = exp6p(data, kw, P, RF)

    R['_meta'] = dict(
        seeds=SEEDS, subsample=SUBSAMPLE, n_modules=N,
        samples=int(data['T']), horizon_hours=float(data['dt'].sum() / 3600),
        fault_classes=ORDER,
        dataset='El Tiemblo Solar+Storage LiFePO4 16s16p, 70 days',
        current_sign_convention='positive = charging (verified on the data)',
        detector_kwargs={k: (v if not isinstance(v, np.floating) else float(v))
                         for k, v in kw.items()},
        iteration=1,
        selected_persistence=P, selected_router_first=bool(RF),
        selection_criterion=R['E12_persistence_selection']['criterion'],
        timestamp=time.strftime('%Y-%m-%dT%H:%M:%S'),
        runtime_seconds=None)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    R['_meta']['runtime_seconds'] = round(time.time() - t0, 1)
    cfg['detector']['router'].update(selected_persistence=int(P),
                                     selected_router_first=bool(RF))
    CONFIG.write_text(json.dumps(cfg, indent=2), encoding='utf-8')
    (OUT_DIR / "revision2_results.json").write_text(
        json.dumps(R, indent=2, default=str), encoding='utf-8')

    agg_path = OUT_DIR / "aggregated_results.json"
    old = json.loads(agg_path.read_text(encoding='utf-8')) if agg_path.exists() else {}
    old['v3'] = R
    agg_path.write_text(json.dumps(old, indent=2, default=str), encoding='utf-8')

    print(f"\nSaved {OUT_DIR/'revision2_results.json'}")
    print(f"Total runtime: {R['_meta']['runtime_seconds']} s "
          f"({R['_meta']['runtime_seconds']/60:.1f} min)")
    return R


if __name__ == '__main__':
    main()
