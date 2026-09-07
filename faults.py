#!/usr/bin/env python3
"""
faults.py -- fault taxonomy for the v3 validation.

Round 1 injected faults by editing the final SOC number a node reported
(+0.15, N(0,0.10), a ramp to +0.20, a flat 0.80).  Applied Energy R1.1 is
right that this cannot test the paper's central claim, and R2 is right that
nothing in it says anything about battery faults.  Here every fault is
injected either on a MEASURED SIGNAL (before the node's estimator runs) or on
a MESSAGE (after it runs), and the taxonomy includes battery faults whose
correct handling is NOT to quarantine the node.

Three families, each with an explicit expected classification:

  node / communication faults (N*)  -> expected: quarantine
  transient glitches          (G*)  -> expected: no quarantine, at most a
                                       transient `suspect`
  battery faults              (B*)  -> expected: no quarantine.  The detector
                                       is not claimed to diagnose them; it is
                                       only claimed not to mistake them for a
                                       misbehaving node.

Signal-level faults (N1, N2, B1-B4) are handled inside PackSignalModel because
they change what the node's own estimator sees.  Message-level faults
(N3-N8, G1-G3) are applied here, to the stream the aggregator receives.

Author: Andrei Garcia Cuadra (ETSIDI-UPM)
"""

import numpy as np

TARGET = 7                      # single-target faults
COLLUSION = [0, 3, 7, 11, 14]   # f = 5
LOSS_NODES = [3, 7, 11]

# name -> (family, expected, level, spec)
CLASSES = {
    # ---- node / communication -------------------------------------------
    'N1_current_offset':  dict(family='node', expected='quarantine', level='signal',
                               kind='i_offset', amp_A=5.0, targets=[TARGET],
                               desc='current-sensor offset +5 A (1 % of full scale)'),
    'N2_voltage_offset':  dict(family='node', expected='quarantine', level='signal',
                               kind='v_offset', amp_V=0.050, targets=[TARGET],
                               desc='voltage-sensor offset +50 mV'),
    'N3_frozen_report':   dict(family='node', expected='quarantine', level='message',
                               kind='stuck', targets=[TARGET],
                               desc='node repeats its last SOC report'),
    'N4_message_corrupt': dict(family='node', expected='quarantine', level='message',
                               kind='corrupt', p=0.05, targets=[TARGET],
                               desc='SOC field replaced by U[0,1] with p=0.05 per message, persistent'),
    'N5_firmware_bias':   dict(family='node', expected='quarantine', level='message',
                               kind='bias', bias=0.05, targets=[TARGET],
                               desc='constant +5 % firmware bias on the report'),
    'N6_collusion':       dict(family='node', expected='quarantine', level='message',
                               kind='collusion', bias=0.03, targets=COLLUSION,
                               desc='5 nodes broadcast true SOC +3 %'),
    'N7_stealth_drift':   dict(family='node', expected='quarantine', level='message',
                               kind='drift', per_step=0.0001, ramp=0.00001, targets=[TARGET],
                               desc='stealthy drift 0.01 % + 0.001 %*k per step'),
    'N8_random_report':   dict(family='node', expected='quarantine', level='message',
                               kind='random', targets=[TARGET],
                               desc='U[0,1] report every step'),
    # ---- transient glitches ---------------------------------------------
    'G1_packet_loss':     dict(family='glitch', expected='no-quarantine', level='message',
                               kind='loss', rate=0.20, burst=(5, 50), targets=LOSS_NODES,
                               from_t0=True,
                               desc='3 honest nodes lose 20 % of messages in bursts of 5-50 steps'),
    'G2_delay':           dict(family='glitch', expected='no-quarantine', level='message',
                               kind='delay', delay=(5, 30), targets=[TARGET],
                               desc='messages arrive 5-30 steps late'),
    'G3_spikes':          dict(family='glitch', expected='no-quarantine', level='message',
                               kind='spike', p=0.01, amp=0.10, targets=[TARGET],
                               desc='1 % of messages carry a single-step +/-10 % spike'),
    # ---- battery faults --------------------------------------------------
    'B1_aging':           dict(family='battery', expected='no-quarantine', level='signal',
                               kind='aging', capacity_loss=0.15, r0_factor=1.5,
                               targets=[TARGET], from_t0=True,
                               desc='15 % capacity fade and 1.5x R0 on one group'),
    'B2_imbalance':       dict(family='battery', expected='no-quarantine', level='signal',
                               kind='imbalance', soc_offset=-0.08, targets=[TARGET],
                               from_t0=True,
                               desc='one group 8 SOC points below the rest'),
    'B3_internal_short':  dict(family='battery', expected='no-quarantine', level='signal',
                               kind='short', r_isc_ohm=33.0, dT_C=0.5, targets=[TARGET],
                               desc='internal short 33 ohm (0.1 A leak, 0.054 %/day at 4480 Ah)'),
    'B3b_internal_short_severe': dict(family='battery', expected='no-quarantine', level='signal',
                               kind='short', r_isc_ohm=2.0625, dT_C=0.5, targets=[TARGET],
                               desc='internal short, 16-parallel-cell equivalent (1.6 A leak, 0.86 %/day)'),
    'B4_thermal':         dict(family='battery', expected='no-quarantine', level='signal',
                               kind='thermal', dT_C=5.0, targets=[TARGET],
                               desc='slow +5 C thermal rise on one group'),
}

ORDER = list(CLASSES.keys())


def physical_fault(name, T, t_start_frac=0.25, **override):
    """Signal-level fault spec consumed by PackSignalModel, or None."""
    spec = CLASSES[name]
    if spec['level'] != 'signal':
        return None
    t0 = 0 if spec.get('from_t0') else int(T * t_start_frac)
    f = {k: v for k, v in spec.items()
         if k not in ('family', 'expected', 'level', 'targets', 'desc', 'from_t0')}
    f.update(override)
    f['target'] = spec['targets'][0]
    f['t_start'] = t0
    return f


def apply_message_faults(build, name, seed, t_start_frac=0.25, **override):
    """Return the stream the aggregator actually receives.

    Output keys
      s_rep  (T,n)  SOC reports as received
      i_rep  (T,n)  current reports as received
      ev     (T,n)  node-declared estimator events (reset / OCV correction)
      src    (T,n)  index of the sample the held message actually carries, i.e.
                    the SEQUENCE NUMBER the RS485 master reads off the frame.
                    src[t,m] == t means a fresh reply; a stalled or receding
                    src is exactly what a bus master observes when a slave
                    misses its poll slot or answers with an old frame.  The
                    detector derives message freshness from this and from
                    nothing else -- no extra sensor is implied.
      stale  (T,n)  legacy flag, kept so the round-2 iteration-0 behaviour
                    remains reproducible; derived from src by the detector now
      t_start, targets
    """
    spec = CLASSES[name]
    rng = np.random.default_rng(seed + 90000)
    T, n = build['T'], build['s_hat'].shape[1]
    t0 = 0 if spec.get('from_t0') else int(T * t_start_frac)
    s = build['s_hat'].copy()
    i = build['I_hat'].copy()
    ev = build['event'].copy()
    stale = np.zeros((T, n), dtype=bool)
    src = np.repeat(np.arange(T)[:, None], n, axis=1)
    tgts = spec['targets']

    if spec['level'] == 'message':
        kind = spec['kind']
        for m in tgts:
            if kind == 'stuck':
                s[t0:, m] = s[max(t0 - 1, 0), m]
                ev[t0:, m] = False
            elif kind == 'corrupt':
                p = override.get('p', spec['p'])
                L = T - t0
                hit = rng.random(L) < p
                draws = rng.random(L)
                # each corruption persists until the next one (forward fill)
                idx = np.where(hit, np.arange(L), -1)
                idx = np.maximum.accumulate(idx)
                val = s[t0:, m].copy()
                live = idx >= 0
                val[live] = draws[idx[live]]
                s[t0:, m] = val
                ev[t0:, m] = False
            elif kind == 'bias':
                s[t0:, m] += override.get('bias', spec['bias'])
            elif kind == 'collusion':
                s[t0:, m] = build['s_true'][t0:, m] + override.get('bias', spec['bias'])
                ev[t0:, m] = False
            elif kind == 'drift':
                k = np.arange(T - t0)
                per = override.get('per_step', spec['per_step'])
                ramp = override.get('ramp', spec['ramp'])
                s[t0:, m] += np.cumsum(per + ramp * k)
            elif kind == 'random':
                s[t0:, m] = rng.random(T - t0)
                ev[t0:, m] = False
            elif kind == 'spike':
                p = override.get('p', spec['p'])
                a = override.get('amp', spec['amp'])
                hit = rng.random(T - t0) < p
                sign = rng.choice([-1.0, 1.0], T - t0)
                s[t0:, m] += hit * sign * a
            elif kind == 'loss':
                rate = override.get('rate', spec['rate'])
                bmin, bmax = spec['burst']
                mean_burst = (bmin + bmax) / 2.0
                # burst-start probability that yields the requested loss rate
                p_start = rate / max((1.0 - rate) * mean_burst, 1e-9)
                lost = np.zeros(T, dtype=bool)
                k = 0
                while k < T:
                    if rng.random() < p_start:
                        L = int(rng.integers(bmin, bmax + 1))
                        lost[k:k + L] = True
                        k += L
                    else:
                        k += 1
                lost[:1] = False
                stale[:, m] |= lost
                # aggregator holds the last received value
                idx = np.where(~lost, np.arange(T), 0)
                idx = np.maximum.accumulate(idx)
                s[:, m] = s[idx, m]
                i[:, m] = i[idx, m]
                ev[:, m] = ev[idx, m]
                src[:, m] = idx
            elif kind == 'delay':
                dmin, dmax = spec['delay']
                lag = rng.integers(dmin, dmax + 1, T)
                srcm = np.maximum(np.arange(T) - lag, 0)
                srcm[:t0] = np.arange(t0)
                s[:, m] = s[srcm, m]
                i[:, m] = i[srcm, m]
                ev[:, m] = ev[srcm, m]
                src[:, m] = srcm
                stale[t0:, m] = True     # timestamp shows the message is late
    s = np.clip(s, 0.0, 1.0)
    return dict(s_rep=s, i_rep=i, ev=ev, stale=stale, src=src, t_start=t0,
                targets=list(tgts), expected=spec['expected'],
                family=spec['family'])


def build_scenario(data, name, seed, lam=1.0, anchor='ocv', t_start_frac=0.25,
                   cfg=None, sig_override=None, msg_override=None,
                   PackSignalModel=None):
    """One full scenario: signal model + physical fault + message fault."""
    from signal_model import PackSignalModel as PSM
    PSM_ = PackSignalModel or PSM
    T_probe = data['T']
    pf = None
    if name is not None:
        pf = physical_fault(name, T_probe, t_start_frac, **(sig_override or {}))
    b = PSM_(data, seed, cfg=cfg, dispersion_scale=lam, anchor=anchor,
             physical_fault=pf).build()
    if name is None:
        T, n = b['T'], b['s_hat'].shape[1]
        b.update(s_rep=b['s_hat'].copy(), i_rep=b['I_hat'].copy(),
                 ev_rx=b['event'].copy(), stale=np.zeros((T, n), dtype=bool),
                 src=np.repeat(np.arange(T)[:, None], n, axis=1),
                 t_start=int(T * t_start_frac), targets=[], expected='none',
                 family='none')
        return b
    mf = apply_message_faults(b, name, seed, t_start_frac, **(msg_override or {}))
    b.update(s_rep=mf['s_rep'], i_rep=mf['i_rep'], ev_rx=mf['ev'],
             stale=mf['stale'], src=mf['src'], t_start=mf['t_start'],
             targets=mf['targets'], expected=mf['expected'],
             family=mf['family'])
    return b
