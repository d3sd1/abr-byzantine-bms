#!/usr/bin/env python3
"""
generate_figures_v3.py -- every round-2 figure, regenerated from results/.

  figR6_fault_class_matrix.pdf     class x outcome heatmap (R1.1, R2)
  figR7_dispersion_sweep.pdf       FPR/TPR vs measured honest dispersion (R1.2)
  figR8_magnitude_vs_spread.pdf    fault magnitude in MAD units (R1.2)
  figR9_aging_vs_byzantine.pdf     aged module vs biased node, time series
  figR1v3_detector_ablation.pdf    ablation on the v3 model
  figR2v3_gap_threshold_sweep.pdf  gap-threshold sweep on the v3 model
  figR5v3_centralised_vs_distributed.pdf

The round-1 figures figR1..figR5 are left untouched as provenance.

Author: Andrei Garcia Cuadra (ETSIDI-UPM)
"""

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams

warnings.filterwarnings('ignore')
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
RES = SCRIPT_DIR / "results" / "revision2_results.json"
FIG = SCRIPT_DIR / "figures"

rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Latin Modern Roman', 'CMU Serif',
                          'Computer Modern Roman', 'DejaVu Serif']
rcParams['mathtext.fontset'] = 'cm'
rcParams['font.size'] = 10
rcParams['axes.labelsize'] = 10
rcParams['axes.titlesize'] = 10
rcParams['legend.fontsize'] = 8
rcParams['xtick.labelsize'] = 8
rcParams['ytick.labelsize'] = 8
rcParams['figure.dpi'] = 200
rcParams['pdf.fonttype'] = 42

CB = {'blue': '#0072B2', 'orange': '#E69F00', 'green': '#009E73',
      'red': '#D55E00', 'purple': '#CC79A7', 'sky': '#56B4E9',
      'yellow': '#F0E442', 'grey': '#999999', 'black': '#000000'}

SHORT = {
    'N1_current_offset': 'N1 current offset +5 A',
    'N2_voltage_offset': 'N2 voltage offset +50 mV',
    'N3_frozen_report': 'N3 frozen report',
    'N4_message_corrupt': 'N4 message corruption',
    'N5_firmware_bias': 'N5 firmware bias +5 %',
    'N6_collusion': 'N6 collusion (5 nodes, +3 %)',
    'N7_stealth_drift': 'N7 stealth drift',
    'N8_random_report': 'N8 random report',
    'G1_packet_loss': 'G1 packet loss (bursty)',
    'G2_delay': 'G2 message delay',
    'G3_spikes': 'G3 single-step spikes',
    'B1_aging': 'B1 aging (-15 % capacity)',
    'B2_imbalance': 'B2 imbalance (-8 SOC pts)',
    'B3_internal_short': 'B3 internal short 33 $\\Omega$',
    'B3b_internal_short_severe': 'B3b internal short (severe)',
    'B4_thermal': 'B4 thermal rise +5 $^\\circ$C',
    'none': 'no fault',
}
FAMILY = {k: ('node' if k.startswith('N') else 'glitch' if k.startswith('G')
              else 'battery' if k.startswith('B') else 'none')
          for k in SHORT}


def load():
    return json.loads(RES.read_text(encoding='utf-8'))


# ----------------------------------------------------------------- figR6
def fig_r6(R):
    e7 = R['E7_fault_class_matrix']
    classes = [c for c in SHORT if c in e7]
    cols = ['quarantined', 'battery-\nanomaly', 'suspect', 'trusted',
            'comms-\ndegraded']
    keys = ['p_target_quarantined', 'p_target_battery',
            'p_target_suspect', 'p_target_trusted', 'p_target_comms_degraded']
    R1 = 'ABR-round-1 (no router, no freshness)'

    def mat(det):
        return np.array([[e7[c][det][k]['mean'] for k in keys] for c in classes])

    C = np.array([[e7[c]['Fixed 3 %']['tpr_steps']['mean'],
                   e7[c]['Fixed 3 %']['fpr_steps']['mean'],
                   e7[c]['Median/MAD z>3']['tpr_steps']['mean'],
                   e7[c]['Median/MAD z>3']['fpr_steps']['mean']]
                  for c in classes])

    fig, axes = plt.subplots(1, 4, figsize=(7.4, 5.3))
    panels = [
        (mat(R1), cols, '(a) round-1 detector\n(no router, no freshness)'),
        (mat('ABR-no-router'), cols, '(b) + message freshness\n(no router)'),
        (mat('ABR-full'), cols, '(c) + physical router\n= v3 final'),
        (C, ['fixed 3 %\ntarget', 'fixed 3 %\nhealthy',
             'Med/MAD\ntarget', 'Med/MAD\nhealthy'],
         '(d) threshold baselines\n(fraction of steps excluded)'),
    ]
    for ax, (M, cl, ttl) in zip(axes, panels):
        im = ax.imshow(M, cmap='YlGnBu', vmin=0, vmax=1, aspect='auto')
        ax.set_xticks(range(len(cl)))
        ax.set_xticklabels(cl, rotation=90, fontsize=6.5)
        ax.set_yticks(range(len(classes)))
        ax.set_title(ttl, fontsize=8)
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                v = M[i, j]
                if v >= 0.005:
                    ax.text(j, i, f'{v:.2f}', ha='center', va='center',
                            fontsize=5.0, color='white' if v > 0.55 else 'black')
        ax.set_xticks(np.arange(-.5, M.shape[1], 1), minor=True)
        ax.set_yticks(np.arange(-.5, len(classes), 1), minor=True)
        ax.grid(which='minor', color='w', linewidth=0.6)
        ax.tick_params(which='minor', length=0)
    axes[0].set_yticklabels([SHORT[c] for c in classes], fontsize=6.5)
    for ax in axes[1:]:
        ax.set_yticklabels([])
    fam = [FAMILY[c] for c in classes]
    for ax in axes:
        for i in range(1, len(fam)):
            if fam[i] != fam[i - 1]:
                ax.axhline(i - 0.5, color='k', lw=1.1)
    fig.colorbar(im, ax=axes, fraction=0.020, pad=0.02,
                 label='probability / fraction of steps')
    fig.savefig(FIG / 'figR6_fault_class_matrix.pdf', bbox_inches='tight')
    plt.close(fig)
    print('  saved figR6_fault_class_matrix.pdf')


# ----------------------------------------------------------------- figR7
def fig_r7(R):
    e8 = R['E8_dispersion_sweep']
    lams = e8['lambdas']
    disp = [e8['dispersion'][str(l)]['iqr']['mean'] * 100 for l in lams]
    dets = ['ABR-full', 'Fixed 3 %', 'Fixed 5 %', 'Median/MAD z>3']
    cols = [CB['blue'], CB['red'], CB['orange'], CB['green']]
    mk = ['o', 's', '^', 'D']
    node_cls = ['N5_firmware_bias', 'N1_current_offset']
    wrong_cls = ['B1_aging', 'G1_packet_loss']

    def curve(cls_list, key):
        return {d: ([np.mean([e8['results'][c][d][str(l)][key]['mean']
                              for c in cls_list]) * 100 for l in lams],
                    [np.mean([e8['results'][c][d][str(l)][key]['std']
                              for c in cls_list]) * 100 for l in lams])
                for d in dets}

    panels = [
        (curve(node_cls, 'tpr_steps'),
         '(a) true node faults detected\n(N5 bias, N1 current offset)',
         'target excluded (% of steps)'),
        (curve(wrong_cls, 'tpr_steps'),
         '(b) battery fault / link glitch\nWRONGLY excluded (B1, G1)',
         'target excluded (% of steps)'),
        (curve(node_cls + wrong_cls, 'fpr_steps'),
         '(c) healthy nodes\nwrongly excluded',
         'healthy excluded (% of steps)'),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.7))
    for ax, (cv, ttl, ylab) in zip(axes, panels):
        for d, c, m in zip(dets, cols, mk):
            y, e = cv[d]
            ax.errorbar(disp, y, yerr=e, marker=m, ms=4, lw=1.3, color=c,
                        capsize=2, label=d)
        ax.set_title(ttl, fontsize=8)
        ax.grid(alpha=0.3)
        ax.set_ylabel(ylab, fontsize=8)
        ax.set_xlabel('honest dispersion: IQR of\n'
                      '$|\\hat{s}_m-\\mathrm{med}_j\\,\\hat{s}_j|$ (% SOC)',
                      fontsize=8)
        ax.set_xlim(0.8, 24)
        ax.set_xscale('log')
        ax.set_xticks([1, 2, 5, 10, 20])
        ax.set_xticklabels(['1', '2', '5', '10', '20'])
    axes[0].set_ylim(-4, 104)
    axes[1].set_ylim(-4, 74)
    axes[2].set_ylim(-4, 74)
    axes[0].legend(fontsize=6.2, loc='upper right', framealpha=0.92)
    fig.tight_layout()
    fig.savefig(FIG / 'figR7_dispersion_sweep.pdf', bbox_inches='tight')
    plt.close(fig)
    print('  saved figR7_dispersion_sweep.pdf')


# ----------------------------------------------------------------- figR8
def fig_r8(R):
    e9 = R['E9_magnitude_vs_spread']
    mad = e9['honest_mad']['mean']
    biases = sorted(float(b) for b in e9['N5_bias'])
    amps = sorted(float(a) for a in e9['N1_current'])
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 2.6))

    x = [b / mad for b in biases]
    for d, c, m in [('ABR-full', CB['blue'], 'o'),
                    ('Fixed 3 %', CB['red'], 's'),
                    ('Median/MAD z>3', CB['green'], 'D')]:
        y = [e9['N5_bias'][str(b)][d]['tpr_steps']['mean'] * 100 for b in biases]
        e = [e9['N5_bias'][str(b)][d]['tpr_steps']['std'] * 100 for b in biases]
        a1.errorbar(x, y, yerr=e, marker=m, ms=4, lw=1.3, color=c, capsize=2, label=d)
    a1.axhline(50, color=CB['grey'], ls=':', lw=0.8)
    a1.set_xlabel('firmware bias / honest cross-node MAD')
    a1.set_ylabel('target excluded (% of steps)')
    a1.set_title('(a) N5 firmware bias', fontsize=9)
    a1.set_ylim(-6, 106)
    a1.grid(alpha=0.3)
    a1.legend(fontsize=6.5, loc='upper left')
    for b, xx in zip(biases, x):
        a1.annotate(f'{b*100:.0f}%', (xx, -8), ha='center', fontsize=6,
                    annotation_clip=False)

    y = [e9['N1_current'][str(a)]['ABR-full']['tpr_steps']['mean'] * 100 for a in amps]
    e = [e9['N1_current'][str(a)]['ABR-full']['tpr_steps']['std'] * 100 for a in amps]
    a2.errorbar(amps, y, yerr=e, marker='o', ms=4, lw=1.3, color=CB['blue'],
                capsize=2, label='ABR-full')
    lat = [e9['N1_current'][str(a)]['ABR-full']['latency_hours']['mean'] for a in amps]
    a2b = a2.twinx()
    a2b.plot(amps, [l if l is not None else np.nan for l in lat], marker='^',
             ms=4, lw=1.3, color=CB['orange'], ls='--', label='latency')
    a2b.set_ylabel('detection latency (h)', color=CB['orange'], fontsize=9)
    a2.set_xlabel('current-sensor offset (A)')
    a2.set_ylabel('target excluded (% of steps)')
    a2.set_title('(b) N1 current-sensor offset', fontsize=9)
    a2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / 'figR8_magnitude_vs_spread.pdf', bbox_inches='tight')
    plt.close(fig)
    print('  saved figR8_magnitude_vs_spread.pdf')


# ----------------------------------------------------------------- figR9
def fig_r9(R):
    """Aged module vs biased node: the figure R1.1 asks for.

    Both panels show the SAME 72-hour window, chosen automatically as the
    window in which the detector first quarantines the biased node, so the
    reader sees the decision being taken.  Rebuilt from the signal model
    rather than from the results JSON, because it needs per-step traces.
    """
    from signal_model import load_pack_data, pack_arrays, DATA_FILE
    from faults import build_scenario
    from abr_detector import ABRDetector

    df = load_pack_data(DATA_FILE, subsample=2)
    data = pack_arrays(df)
    kw = dict(R['_meta']['detector_kwargs'])
    for k in ('th_ci', 'th_cusum', 'cusum_slack', 'mad_floor', 'alpha_up',
              'alpha_down', 'ewma_cap', 'sigmoid_k', 'sigmoid_x0', 'trust_high',
              'trust_low', 'trust_battery', 'gap_threshold',
              'min_ewma_quarantine', 'suspect_z', 'ci_alpha', 'ci_mad_floor',
              'eta', 'q_nom', 'deadband'):
        kw[k] = float(kw[k])
    kw['n'] = int(kw['n'])
    kw['f_max'] = int(kw['f_max'])

    tgt = 7
    sc_b1 = build_scenario(data, 'B1_aging', 42)
    sc_n5 = build_scenario(data, 'N5_firmware_bias', 42)
    r_b1 = ABRDetector(router=True, **kw).run(sc_b1)
    r_n5 = ABRDetector(router=True, **kw).run(sc_n5)

    hrs = np.cumsum(data['dt']) / 3600.0
    ts = sc_n5['t_start']
    ev = int(r_n5['first_ex'][tgt])
    centre = hrs[ev] if ev >= 0 else hrs[ts] + 48.0
    t_lo = max(centre - 36.0, hrs[0])
    t_hi = t_lo + 72.0
    m = (hrs >= t_lo) & (hrs <= t_hi)
    x = hrs[m] - t_lo

    fig, axes = plt.subplots(2, 1, figsize=(7.2, 4.6), sharex=True)
    for ax, sc, r, ttl, lab in [
            (axes[0], sc_b1, r_b1,
             '(a) B1: healthy-but-aged module (15 % capacity fade)',
             'aged module report'),
            (axes[1], sc_n5, r_n5,
             '(b) N5: node broadcasting a +5 % firmware bias',
             'biased node report')]:
        med = np.median(sc['s_rep'], axis=1)
        q = r['excl'][:, tgt][m]
        b = r['batt_hist'][:, tgt][m]
        if q.any():
            ax.fill_between(x, -5, 105, where=q, color=CB['red'], alpha=0.16,
                            step='mid', lw=0, label='quarantined by detector')
        if b.any():
            ax.fill_between(x, -5, 105, where=b, color=CB['sky'], alpha=0.22,
                            step='mid', lw=0, label='battery-anomaly flag')
        for j in range(sc['s_rep'].shape[1]):
            if j != tgt:
                ax.plot(x, sc['s_rep'][m, j] * 100, color=CB['grey'], lw=0.4,
                        alpha=0.45, zorder=2)
        ax.plot([], [], color=CB['grey'], lw=0.8, alpha=0.7,
                label='15 honest node reports')
        ax.plot(x, med[m] * 100, color=CB['black'], lw=1.3, ls='--',
                label='median of reports', zorder=5)
        ax.plot(x, sc['soc_pack_true'][m] * 100, color=CB['green'], lw=1.3,
                label='true pack SOC', zorder=5)
        ax.plot(x, sc['s_true'][m, tgt] * 100, color=CB['orange'], lw=1.2,
                ls=':', label='true SOC of that module', zorder=6)
        ax.plot(x, sc['s_rep'][m, tgt] * 100, color=CB['red'], lw=1.5,
                label=lab, zorder=7)
        if t_lo <= hrs[ts] <= t_hi:
            ax.axvline(hrs[ts] - t_lo, color=CB['purple'], lw=1.1, ls='-.')
        ax.set_title(ttl, fontsize=9)
        ax.set_ylabel('SOC (%)')
        ax.grid(alpha=0.25)
        lo = min(sc['s_rep'][m].min(), sc['soc_pack_true'][m].min()) * 100
        hi = max(sc['s_rep'][m].max(), sc['soc_pack_true'][m].max()) * 100
        pad = max(0.12 * (hi - lo), 1.5)
        ax.set_ylim(lo - pad, hi + pad + 0.30 * (hi - lo))
        ax.legend(fontsize=6, ncol=3, loc='upper left', framealpha=0.92)
    axes[1].set_xlabel('time (h) within a 72-hour window centred on the '
                       'quarantine decision')
    fig.tight_layout()
    fig.savefig(FIG / 'figR9_aging_vs_byzantine.pdf', bbox_inches='tight')
    plt.close(fig)
    print('  saved figR9_aging_vs_byzantine.pdf')


# --------------------------------------------------------------- figR1v3
def fig_r1v3(R):
    e1 = R['E1p_ablation']
    variants = ['EWMA trust only', 'Gap only (no trust grading)',
                'EWMA + gap (round-1 detector)',
                'EWMA + gap + message freshness',
                'EWMA + gap + router (v3)']
    short = ['EWMA only', 'gap only', 'EWMA+gap\n(round 1)',
             '+ message\nfreshness', '+ router\n(v3 final)']
    node = [c for c in e1 if not c.startswith('_')]
    fp = e1['_false_positive_check']
    fpc = list(fp.keys())
    cols = [CB['grey'], CB['orange'], CB['red'], CB['purple'], CB['blue']]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 3.2))
    xn = np.arange(len(node)); w = 0.16
    for i, v in enumerate(variants):
        y = [e1[c][v]['tpr_steps']['mean'] * 100 for c in node]
        e = [e1[c][v]['tpr_steps']['std'] * 100 for c in node]
        a1.bar(xn + i * w, y, w, yerr=e, color=cols[i], capsize=1.5,
               label=short[i], alpha=0.92)
    a1.set_xticks(xn + 2.0 * w)
    a1.set_xticklabels([SHORT[c].split('(')[0].strip() for c in node],
                       rotation=18, ha='right', fontsize=6.5)
    a1.set_ylabel('node fault excluded (% of steps)', fontsize=9)
    a1.set_title('(a) detection of true node faults', fontsize=9)
    a1.set_ylim(0, 105)
    a1.grid(alpha=0.3, axis='y')
    a1.legend(fontsize=6, ncol=3, loc='upper center',
              bbox_to_anchor=(0.5, -0.30), framealpha=0.92)

    xf = np.arange(len(fpc))
    for i, v in enumerate(variants):
        y = [fp[c][v]['p_target_quarantined']['mean'] * 100 for c in fpc]
        e = [fp[c][v]['p_target_quarantined']['std'] * 100 for c in fpc]
        a2.bar(xf + i * w, y, w, yerr=e, color=cols[i], capsize=1.5, alpha=0.92)
    a2.set_xticks(xf + 2.0 * w)
    a2.set_xticklabels([SHORT[c].split('(')[0].strip() for c in fpc],
                       rotation=18, ha='right', fontsize=6.5)
    a2.set_ylabel('wrongly quarantined (% of seeds)', fontsize=9)
    a2.set_title('(b) glitches and battery faults\n(any quarantine is an error)',
                 fontsize=9)
    a2.grid(alpha=0.3, axis='y')
    fig.tight_layout()
    fig.savefig(FIG / 'figR1v3_detector_ablation.pdf', bbox_inches='tight')
    plt.close(fig)
    print('  saved figR1v3_detector_ablation.pdf')


# --------------------------------------------------------------- figR2v3
def fig_r2v3(R):
    e2 = R['E2p_gap_sweep']
    g = e2['gaps']
    s = e2['sweep']
    fig, ax = plt.subplots(figsize=(3.6, 2.7))
    tpr = [s[str(x)]['q_node']['mean'] * 100 for x in g]
    fpr = [s[str(x)]['q_batt']['mean'] * 100 for x in g]
    hfp = [s[str(x)]['fpr']['mean'] * 100 for x in g]
    ax.plot(g, tpr, marker='o', ms=4, color=CB['blue'],
            label='P(quarantine $\\mid$ node fault)')
    ax.plot(g, fpr, marker='s', ms=4, color=CB['red'],
            label='P(quarantine $\\mid$ glitch/battery)')
    ax.plot(g, hfp, marker='^', ms=4, color=CB['green'],
            label='healthy nodes excluded (% steps)')
    ax.set_xlabel('gap-quarantine threshold $\\delta$')
    ax.set_ylabel('%')
    ax.grid(alpha=0.3)
    ax.legend(fontsize=6.5)
    fig.tight_layout()
    fig.savefig(FIG / 'figR2v3_gap_threshold_sweep.pdf', bbox_inches='tight')
    plt.close(fig)
    print('  saved figR2v3_gap_threshold_sweep.pdf')


# --------------------------------------------------------------- figR5v3
def fig_r5v3(R):
    e6 = R['E6p_centralised']
    cls = list(e6.keys())
    fig, ax = plt.subplots(figsize=(4.2, 2.7))
    x = np.arange(len(cls)); w = 0.35
    c = [e6[k]['Centralised mean']['rmse']['mean'] * 100 for k in cls]
    cs = [e6[k]['Centralised mean']['rmse']['std'] * 100 for k in cls]
    d = [e6[k]['Distributed ABR']['rmse']['mean'] * 100 for k in cls]
    ds = [e6[k]['Distributed ABR']['rmse']['std'] * 100 for k in cls]
    ax.bar(x - w / 2, c, w, yerr=cs, color=CB['red'], capsize=2,
           label='centralised mean fusion', alpha=0.92)
    ax.bar(x + w / 2, d, w, yerr=ds, color=CB['blue'], capsize=2,
           label='distributed ABR (v3)', alpha=0.92)
    ax.set_yscale('log')
    ax.set_xticks(x)
    ax.set_xticklabels([SHORT[k].split('(')[0].strip() for k in cls],
                       rotation=18, ha='right', fontsize=7)
    ax.set_ylabel('consensus SOC RMSE (%, log)')
    ax.grid(alpha=0.3, axis='y', which='both')
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / 'figR5v3_centralised_vs_distributed.pdf', bbox_inches='tight')
    plt.close(fig)
    print('  saved figR5v3_centralised_vs_distributed.pdf')


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    R = load()
    rcParams['text.usetex'] = False
    fig_r6(R)
    fig_r7(R)
    fig_r8(R)
    fig_r1v3(R)
    fig_r2v3(R)
    fig_r5v3(R)
    fig_r9(R)
    print('All v3 figures written to', FIG)


if __name__ == '__main__':
    main()
