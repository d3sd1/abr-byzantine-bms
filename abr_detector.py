#!/usr/bin/env python3
"""
abr_detector.py -- ABR statistical fault detector + physical-consistency router.

Clean reimplementation of the detector of `run_evaluation.ABR_PINN` with every
reference to a PINN or to `physics_models` removed: the detector never was
physics-informed in any neural sense, it is robust statistics plus a quarantine
rule, and the paper is repositioned accordingly.

NEW in v3: Step 2e, the physical-consistency router.  The round-1 detector
quarantined any node that crossed the gap criterion, which is what Applied
Energy R1.1 objects to -- a healthy-but-aged or imbalanced module that
legitimately drifts away from its peers would be treated as misbehaving.  The
router decides, for each node that crosses the gap, whether the divergence is a
NODE fault (quarantine) or a physically consistent BATTERY divergence (flag,
down-weight, keep in the consensus), from two elementary properties of a single
series string:

  (i)  shared-current consistency.  Every node sits on the same string and must
       measure the same current:
           c_I,m = |I^_m - median_j I^_j| / (1.4826 MAD_j),  EWMA-smoothed.
       A node whose current disagrees with the string is a sensor/node fault.

  (ii) report-vs-current self-consistency.  A node's reported SOC increment must
       match what its own transmitted current implies:
           r_m[k] = ds^_m[k] - ds_pack[k] - eta (I^_m - I_bus) dt / (3600 Q),
       accumulated in a two-sided CUSUM.  Steps the node itself flags as
       estimator events (OCV correction, full-charge resync) and steps the
       aggregator knows are stale are excluded.  A stuck, corrupted, biased or
       drifting node fails this test; an aged or imbalanced module passes it,
       because it reports its own Coulomb count faithfully -- its capacity is
       simply not what the counter assumes.

  Rule: gap crossed AND (current inconsistent OR CUSUM inconsistent)
            -> quarantine
        gap crossed AND both consistent
            -> battery-anomaly (trust reduced, not quarantined)

Thresholds are calibrated from FAULT-FREE runs over the first 10 % of the
horizon and then frozen; they are never set by looking at the fault classes.

Author: Andrei Garcia Cuadra (ETSIDI-UPM)
"""

import numpy as np

TRUSTED, SUSPECT, QUARANTINED, BATTERY = 0, 1, 2, 3
STATE_NAMES = ['trusted', 'suspect', 'quarantined', 'battery-anomaly']
COMMS_DEGRADED = 'comms-degraded'


def message_freshness(sc, max_age=2):
    """Which held messages the RS485 master knows to be stale.

    Derived only from the sequence number the master already reads off every
    frame: a reply is fresh if its sequence number advanced since the previous
    poll and its age does not exceed `max_age` poll cycles.  No extra sensor
    and no extra traffic is implied -- any bus master has exactly this.
    """
    T, n = sc['s_rep'].shape
    src = sc.get('src')
    if src is None:                      # fall back to the legacy flag
        deg = sc['stale'].copy()
        deg[0] = False
        return deg
    advanced = np.diff(src, axis=0, prepend=src[:1] - 1) > 0
    age = np.arange(T)[:, None] - src
    deg = (~advanced) | (age > max_age)
    deg[0] = False
    return deg


# ----------------------------------------------------------------------
# Router statistics -- state-independent, so computed once, vectorised
# ----------------------------------------------------------------------
def router_stats(sc, *, ci_alpha=0.05, ci_mad_floor=0.20, cusum_slack=9.7e-4,
                 eta=0.995, q_nom=4480.0, deadband=2.0, degraded=None):
    """Return (ewma_ci, cusum) arrays of shape (T,n).

    Neither statistic depends on the detector's quarantine state, so both can
    be produced in closed form.  The one-sided CUSUM with reset at zero,
    S[t] = max(0, S[t-1] + r[t] - k), equals C[t] - min(0, min_{j<=t} C[j])
    with C = cumsum(r - k), which is exact and fully vectorised.
    """
    i_rep, s_rep = sc['i_rep'], sc['s_rep']
    ev, stale, dt = sc['ev_rx'], sc['stale'], sc['dt']
    soc_bms, I_bus = sc['soc_bms'], sc['I_pack']
    T, n = s_rep.shape
    # The first FRESH sample after a staleness gap carries the change of the
    # whole gap, which the aggregator cannot attribute to one step, so it is
    # excluded from the model-based tests along with the stale samples
    # themselves.  Without this a burst of packet loss looks like a node fault.
    deg = stale if degraded is None else degraded
    blind = deg | np.roll(deg, 1, axis=0)
    blind[0] = True

    # (i) shared-current consistency
    mi = np.median(i_rep, axis=1, keepdims=True)
    mad = np.maximum(np.median(np.abs(i_rep - mi), axis=1, keepdims=True) * 1.4826,
                     ci_mad_floor)
    ci = np.abs(i_rep - mi) / mad
    ci = np.where(blind, 0.0, ci)          # no update on known-stale messages
    ewma_ci = np.zeros((T, n))
    acc = np.zeros(n)
    a = ci_alpha
    for t in range(T):
        upd = ~blind[t]
        acc = np.where(upd, (1 - a) * acc + a * ci[t], acc)
        ewma_ci[t] = acc

    # (ii) report-vs-current self-consistency
    i_eff = np.where(np.abs(i_rep) < deadband, 0.0, i_rep)
    ib = np.where(np.abs(I_bus) < deadband, 0.0, I_bus)
    ds_pack = np.diff(soc_bms, prepend=soc_bms[0])
    exp_d = ds_pack[:, None] + eta * (i_eff - ib[:, None]) * dt[:, None] / (3600.0 * q_nom)
    d_rep = np.diff(s_rep, axis=0, prepend=s_rep[:1])
    r = d_rep - exp_d
    r[0] = 0.0
    r = np.where(ev | blind, 0.0, r)
    Cp = np.cumsum(r - cusum_slack, axis=0)
    Sp = Cp - np.minimum.accumulate(np.minimum(Cp, 0.0), axis=0)
    Cn = np.cumsum(-r - cusum_slack, axis=0)
    Sn = Cn - np.minimum.accumulate(np.minimum(Cn, 0.0), axis=0)
    return ewma_ci, np.maximum(Sp, Sn)


class ABRDetector:
    """Trust-weighted Byzantine-resilient consensus with physical routing."""

    name = 'ABR-full'

    def __init__(self, n=16, f_max=5, *, mad_floor=0.005,
                 alpha_up=0.10, alpha_down=0.30, ewma_cap=25.0,
                 sigmoid_k=1.5, sigmoid_x0=6.0,
                 trust_high=0.8, trust_low=0.4, trust_battery=0.1,
                 gap_threshold=3.0, min_ewma_quarantine=5.0, suspect_z=3.0,
                 router=True, trust_grading=True, gap_quarantine=True,
                 router_first=False,
                 ci_alpha=0.05, ci_mad_floor=0.20,
                 th_ci=3.85, th_cusum=6.6e-5, cusum_slack=9.7e-4,
                 cusum_persistence=1, comms_state=True, comms_max_age=2,
                 eta=0.995, q_nom=4480.0, deadband=2.0, latch=True):
        self.n, self.f = n, f_max
        self.mad_floor = mad_floor
        self.a_up, self.a_dn, self.cap = alpha_up, alpha_down, ewma_cap
        self.k, self.x0 = sigmoid_k, sigmoid_x0
        self.gh, self.gl, self.gb = trust_high, trust_low, trust_battery
        self.gap_th, self.min_ewma = gap_threshold, min_ewma_quarantine
        self.suspect_z = suspect_z
        self.use_router, self.grading, self.quar = router, trust_grading, gap_quarantine
        self.ci_alpha, self.ci_floor = ci_alpha, ci_mad_floor
        self.th_ci, self.th_cusum, self.slack = th_ci, th_cusum, cusum_slack
        self.eta, self.q, self.db, self.latch = eta, q_nom, deadband, latch
        self.router_first = router_first
        self.persist = max(int(cusum_persistence), 1)
        self.comms_state, self.comms_max_age = comms_state, int(comms_max_age)
        if not router:
            self.name = 'ABR-no-router'
        elif router_first:
            self.name = 'ABR-router-first'
        if router and self.persist > 1:
            self.name += f' (P={self.persist})'

    # ------------------------------------------------------------------
    def run(self, sc):
        n, f = self.n, self.f
        s_rep = sc['s_rep']
        T = s_rep.shape[0]
        degraded = (message_freshness(sc, self.comms_max_age)
                    if self.comms_state else np.zeros(s_rep.shape, bool))
        ewma_ci, cusum = router_stats(
            sc, ci_alpha=self.ci_alpha, ci_mad_floor=self.ci_floor,
            cusum_slack=self.slack, eta=self.eta, q_nom=self.q,
            deadband=self.db, degraded=degraded if self.comms_state else None)
        bad_i = ewma_ci > self.th_ci
        raw_c = cusum > self.th_cusum
        if self.persist > 1:
            # A single-sample spike puts +d then -d into the residual, so the
            # CUSUM returns to baseline; a step change (firmware bias,
            # collusion) keeps it above the threshold.  Requiring the excursion
            # to last P consecutive polls therefore separates spike from step
            # without looking at the fault taxonomy.
            run = np.zeros(raw_c.shape, dtype=np.int32)
            acc = np.zeros(raw_c.shape[1], dtype=np.int32)
            for t in range(raw_c.shape[0]):
                acc = np.where(raw_c[t], acc + 1, 0)
                run[t] = acc
            bad_c = run >= self.persist
        else:
            bad_c = raw_c
        if self.latch:
            bad_i = np.maximum.accumulate(bad_i, axis=0)
            bad_c = np.maximum.accumulate(bad_c, axis=0)

        quar = np.zeros(n, bool)
        batt = np.zeros(n, bool)
        ew = np.zeros(n)
        cons = np.zeros(T)
        excl = np.zeros((T, n), bool)
        batt_hist = np.zeros((T, n), bool)
        ever_suspect = np.zeros(n, bool)
        ever_comms = np.zeros(n, bool)
        first_flag = np.full(n, -1)
        a_up, a_dn, cap = self.a_up, self.a_dn, self.cap
        gl, gh, gb, kk, x0 = self.gl, self.gh, self.gb, self.k, self.x0
        floor, gap_th, min_ew = self.mad_floor, self.gap_th, self.min_ewma
        sus_z, use_router, grading, do_quar = (self.suspect_z, self.use_router,
                                               self.grading, self.quar)
        nq = 0
        t = 0
        while t < T:
            # The active set only changes when a node is quarantined (at most f
            # times), so the robust statistics are computed once per segment,
            # vectorised over the remaining horizon, instead of once per step.
            act = ~quar
            seg = s_rep[t:]
            ar = seg[:, act]
            med = np.median(ar, axis=1, keepdims=True)
            mad = np.median(np.abs(ar - med), axis=1, keepdims=True)
            np.maximum(mad, floor, out=mad)
            zseg = np.abs(seg - med) / (1.4826 * mad)
            restart = False
            for k in range(zseg.shape[0]):
                tt = t + k
                z = zseg[k]
                dg = degraded[tt]
                a = np.where(z >= ew, a_up, a_dn)
                ew_new = (1.0 - a) * ew + a * z
                np.minimum(ew_new, cap, out=ew_new)
                # a node the master knows to be stale accrues no evidence:
                # its EWMA is frozen and its report carries no weight, but its
                # trust is untouched and it recovers as soon as frames arrive
                ew = np.where(act & ~dg, ew_new, ew)
                ever_suspect |= ew > sus_z
                ever_comms |= dg

                if grading:
                    trust = gl + (gh - gl) / (1.0 + np.exp(kk * (ew - x0)))
                else:
                    trust = np.full(n, gh)
                if batt.any():
                    trust = np.where(batt, gb, trust)

                # Escalation: a node already flagged as a battery anomaly is
                # re-classified as a node fault as soon as router evidence
                # arrives, so the gap/router latency race cannot hide a fault.
                # With router_first the same evidence also quarantines nodes
                # that never crossed the gap at all (the collusion case).
                pool = (~quar) if self.router_first else batt
                if use_router and pool.any() and nq < f:
                    esc = np.flatnonzero(pool & ~dg & (bad_i[tt] | bad_c[tt]))
                    for node in esc:
                        if nq < f and not quar[node]:
                            batt[node] = False
                            quar[node] = True
                            nq += 1
                            restart = True
                            if first_flag[node] < 0:
                                first_flag[node] = tt
                    if restart:
                        excl[tt] = quar
                        batt_hist[tt] = batt
                        w = np.where(quar | dg, 0.0, trust)
                        ws = w.sum()
                        cons[tt] = (np.median(s_rep[tt]) if ws < 1e-10
                                    else float(np.dot(w / ws, s_rep[tt])))
                        t = tt + 1
                        break

                # gap-based quarantine (pre-check avoids a sort at every step)
                if do_quar and nq < f and ew.max() > min_ew:
                    cand = np.flatnonzero(act & ~batt & ~dg)
                    if len(cand) > f + 1:
                        ewc = ew[cand]
                        order = np.argsort(-ewc)
                        se = ewc[order]
                        mg, gp = 0.0, -1
                        for j in range(min(f, len(se) - 1)):
                            g = se[j] - se[j + 1]
                            if g > mg:
                                mg, gp = g, j
                        if mg > gap_th and se[0] > min_ew:
                            for j in range(gp + 1):
                                node = cand[order[j]]
                                if quar[node] or batt[node]:
                                    continue
                                if (not use_router) or bad_i[tt, node] or bad_c[tt, node]:
                                    if nq < f:
                                        quar[node] = True
                                        nq += 1
                                        restart = True
                                        if first_flag[node] < 0:
                                            first_flag[node] = tt
                                else:
                                    batt[node] = True
                                    trust[node] = gb
                                    if first_flag[node] < 0:
                                        first_flag[node] = tt

                w = np.where(quar | dg, 0.0, trust)
                ws = w.sum()
                cons[tt] = (np.median(s_rep[tt]) if ws < 1e-10
                            else float(np.dot(w / ws, s_rep[tt])))
                excl[tt] = quar
                batt_hist[tt] = batt
                if restart:
                    t = tt + 1
                    break
            else:
                t = T

        worst = np.where(ever_suspect, SUSPECT, TRUSTED)
        worst = np.where(batt_hist.any(axis=0), BATTERY, worst)
        worst = np.where(excl.any(axis=0), QUARANTINED, worst)
        first_ex = np.full(n, -1)
        for m in range(n):
            w_ = np.flatnonzero(excl[:, m])
            if w_.size:
                first_ex[m] = int(w_[0])
        return dict(cons=cons, excl=excl, worst=worst, first_ex=first_ex,
                    quarantined=quar.copy(), battery=batt.copy(),
                    batt_hist=batt_hist, first_flag=first_flag,
                    ewma_ci=ewma_ci, cusum=cusum, bad_i=bad_i, bad_c=bad_c,
                    degraded=degraded, ever_comms=ever_comms)


# ======================================================================
# Baselines -- identical signal model, identical fault injection.
# All are memoryless, so they are evaluated in closed form over the horizon.
# ======================================================================

class _Memoryless:
    def run(self, sc):
        rep = sc['s_rep']
        T, n = rep.shape
        excl = self.exclude(rep)
        keep = ~excl
        cnt = keep.sum(axis=1)
        tot = np.where(keep, rep, 0.0).sum(axis=1)
        med = np.median(rep, axis=1)
        cons = np.where(cnt > 0, tot / np.maximum(cnt, 1), med)
        worst = np.where(excl.any(axis=0), QUARANTINED, TRUSTED)
        first_ex = np.full(n, -1)
        for m in range(n):
            w_ = np.flatnonzero(excl[:, m])
            if w_.size:
                first_ex[m] = int(w_[0])
        return dict(cons=cons, excl=excl, worst=worst, first_ex=first_ex,
                    quarantined=excl[-1].copy(), battery=np.zeros(n, bool),
                    first_flag=first_ex)


class MedianMADDetector(_Memoryless):
    """Instantaneous robust Hampel filter: flag |x-med|/(1.4826 MAD) > 3."""
    name = 'Median/MAD z>3'

    def __init__(self, n=16, f_max=5, thresh=3.0, mad_floor=0.005):
        self.n, self.f, self.th, self.floor = n, f_max, thresh, mad_floor

    def exclude(self, rep):
        med = np.median(rep, axis=1, keepdims=True)
        dev = np.abs(rep - med)
        mad = np.maximum(np.median(dev, axis=1, keepdims=True), self.floor)
        return dev / (1.4826 * mad) > self.th


class FixedThresholdDetector(_Memoryless):
    """The trivial 'easy by physics' comparator R1.2 asks for.

    Exclude any node whose report differs from the median by more than a fixed
    absolute SOC margin.  No statistics, no memory, no tuning.
    """

    def __init__(self, n=16, f_max=5, margin=0.03):
        self.n, self.f, self.margin = n, f_max, margin
        self.name = 'Fixed |dev|>%d %%' % round(margin * 100)

    def exclude(self, rep):
        med = np.median(rep, axis=1, keepdims=True)
        return np.abs(rep - med) > self.margin


class PBFTTrimmedMean(_Memoryless):
    """PBFT-BMS style: discard the f highest and f lowest, average the rest."""
    name = 'PBFT-BMS trimmed mean'

    def __init__(self, n=16, f_max=5):
        self.n, self.f = n, f_max

    def exclude(self, rep):
        T, n = rep.shape
        excl = np.zeros((T, n), bool)
        if 2 * self.f < n:
            order = np.argsort(rep, axis=1)
            rows = np.arange(T)[:, None]
            excl[rows, order[:, :self.f]] = True
            excl[rows, order[:, -self.f:]] = True
        return excl


class CentralisedMean(_Memoryless):
    """Centralised fusion with no detection at all (mean of every report)."""
    name = 'Centralised mean (no detection)'

    def __init__(self, n=16, f_max=5):
        self.n, self.f = n, f_max

    def exclude(self, rep):
        return np.zeros(rep.shape, bool)


def run_detector(det, sc):
    return det.run(sc)
