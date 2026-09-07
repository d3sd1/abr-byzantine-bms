#!/usr/bin/env python3
"""
Graphical abstract, round 2.

Every reference to a PINN is gone: the method is a statistical fault detector
with a physical-consistency router.  The flow shown is the one the paper now
claims and the experiments now test:

  measured signals -> local estimator -> statistical detector
                   -> router (node fault vs battery divergence) -> consensus

Numbers in the results banner are read from results/revision2_results.json,
so the figure cannot drift away from the experiments.

Output: graphical_abstract.pdf / .png, written next to paper.tex when
this script runs inside the manuscript tree, otherwise beside the script.
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle, Rectangle
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
RES = SCRIPT_DIR / "results" / "revision2_results.json"
# Inside the manuscript tree the abstract belongs next to paper.tex; in a bare
# checkout of the reproducibility repo it is written beside the script.
OUT = (SCRIPT_DIR.parent if (SCRIPT_DIR.parent / "paper.tex").exists()
       else SCRIPT_DIR)

DEEP_BLUE = "#1a5276"
LIGHT_BLUE = "#2980b9"
VERY_LIGHT_BLUE = "#d6eaf8"
GREEN = "#1e8449"
GREEN_L = "#d5f5e3"
AMBER = "#f39c12"
AMBER_L = "#fdebd0"
RED = "#c0392b"
RED_L = "#fadbd8"
WHITE = "#ffffff"
DARK = "#2c3e50"
MED = "#7f8c8d"
PANEL = "#f8f9fa"

FW, FH = 14.0, 6.0
fig, ax = plt.subplots(figsize=(FW, FH), dpi=300)
ax.set_xlim(0, FW); ax.set_ylim(0, FH); ax.set_aspect("equal"); ax.axis("off")
fig.patch.set_facecolor(WHITE)


def rbox(xy, w, h, fc, ec=None, lw=1.5, alpha=1.0, z=2, pad=0.06):
    ax.add_patch(FancyBboxPatch(xy, w, h, boxstyle=f"round,pad={pad}",
                                fc=fc, ec=(ec or fc), lw=lw, alpha=alpha, zorder=z))


def txt(x, y, s, fs=8, fw="normal", fc=DARK, ha="center", va="center", z=10, **kw):
    ax.text(x, y, s, fontsize=fs, fontweight=fw, color=fc, ha=ha, va=va,
            fontfamily="serif", zorder=z, **kw)


def arr(x0, y0, x1, y1, color=DEEP_BLUE, lw=3.0, ms=20, style="-|>"):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle=style, color=color, lw=lw,
                                mutation_scale=ms), zorder=12)


M = 0.20
TY, TH = 5.18, 0.68
BY, BH = 0.08, 0.98
PY = BY + BH + 0.14
PH = TY - PY - 0.12
TW = FW - 2 * M

p1w = TW * 0.26; p1x = M
p2w = TW * 0.27; p2x = p1x + p1w
p3w = TW * 0.25; p3x = p2x + p2w
p4w = TW * 0.22; p4x = p3x + p3w

# ---------------------------------------------------------------- title
rbox((M, TY), TW, TH, DEEP_BLUE, lw=0, pad=0.08)
txt(FW / 2, TY + TH - 0.24,
    "Byzantine-Resilient Consensus for Distributed BMS: "
    "separating node faults from battery faults",
    fs=13.0, fw="bold", fc=WHITE)
txt(FW / 2, TY + 0.19,
    "16 nodes  ·  70 days of real LiFePO$_4$ field data  ·  "
    "16 fault classes injected on measured signals  ·  5 seeds",
    fs=8.5, fc="#aed6f1")

for px, pw, num, sub in [
    (p1x, p1w, "1", "Measured signals → local estimator"),
    (p2x, p2w, "2", "Statistical detector"),
    (p3x, p3w, "3", "Physical-consistency router"),
    (p4x, p4w, "4", "Trust-weighted consensus"),
]:
    rbox((px, PY), pw, PH, PANEL, ec="#d5d8dc", lw=1.0, alpha=0.6)
    rbox((px, PY + PH - 0.52), pw, 0.52, DEEP_BLUE, lw=0, alpha=0.10, pad=0.0)
    txt(px + pw / 2, PY + PH - 0.17, f"Step {num}", fs=11.5, fw="bold", fc=DEEP_BLUE)
    txt(px + pw / 2, PY + PH - 0.40, sub, fs=7.5, fc=MED)

# ================================================== Step 1: signals
c1 = p1x + p1w / 2
bw, bh, dx, dy = 0.26, 0.30, 0.34, 0.40
g_y0 = PY + PH - 0.95
for r in range(4):
    for c in range(4):
        cx = c1 - 1.5 * dx + c * dx
        cy = g_y0 - r * dy
        rbox((cx - bw / 2, cy - bh / 2), bw, bh, LIGHT_BLUE, ec=DEEP_BLUE,
             lw=0.6, alpha=0.85, z=3, pad=0.03)
        txt(cx, cy, str(r * 4 + c + 1), fs=5, fw="bold", fc=WHITE, z=5)
txt(c1, g_y0 + bh / 2 + 0.13,
    "16 series groups · one BMS node each", fs=7, fc=MED)
txt(c1, g_y0 - 3 * dy - bh / 2 - 0.22,
    r"$\hat{V}_m,\ \hat{I}_m,\ \hat{T}_m$" + "   (real 70-day field log)",
    fs=8, fw="bold", fc=DEEP_BLUE)

ew, eh = p1w * 0.84, 0.72
ex = p1x + (p1w - ew) / 2
ey = PY + 0.22
rbox((ex, ey), ew, eh, VERY_LIGHT_BLUE, ec=LIGHT_BLUE, lw=1.4)
txt(ex + ew / 2, ey + eh - 0.20, "local estimator", fs=9, fw="bold", fc=DEEP_BLUE)
txt(ex + ew / 2, ey + eh - 0.44,
    r"differential $\int\!(\hat{I}_m-I_{\rm bus})\,dt$", fs=7.5, fc=DARK)
txt(ex + ew / 2, ey + 0.14, r"broadcast $\hat{s}_m$ + status flags",
    fs=7, fc=MED)
arr(c1, g_y0 - 3 * dy - bh / 2 - 0.40, c1, ey + eh + 0.05, color=MED, lw=1.8, ms=14)

# ================================================== Step 2: detector
c2 = p2x + p2w / 2
sw, sh = p2w * 0.86, 0.40
sy0 = PY + PH - 1.05
for i, lab in enumerate([r"robust $z$-score (median / MAD)",
                         "asymmetric EWMA trust",
                         r"gap rule $\Rightarrow$ candidate"]):
    sy = sy0 - i * 0.60
    rbox((c2 - sw / 2, sy - sh / 2), sw, sh, WHITE, ec=DEEP_BLUE, lw=1.2, z=6, pad=0.04)
    txt(c2, sy, lab, fs=8, fw="bold", fc=DEEP_BLUE, z=7)
    if i < 2:
        arr(c2, sy - sh / 2 - 0.02, c2, sy - 0.60 + sh / 2 + 0.02,
            color=LIGHT_BLUE, lw=1.6, ms=12)

# honest dispersion illustration
hx, hy, hw, hh = c2 - sw / 2, PY + 0.24, sw, 1.05
rbox((hx, hy), hw, hh, WHITE, ec="#d5d8dc", lw=1.0)
rng = np.random.default_rng(7)
t = np.linspace(0, 1, 120)
base = 0.5 + 0.16 * np.sin(2 * np.pi * t)
for k in range(14):
    y = base + rng.normal(0, 0.035) + 0.02 * rng.normal(0, 1, 120).cumsum() / 40
    ax.plot(hx + 0.14 + t * (hw - 0.28), hy + 0.20 + y * (hh - 0.42),
            color=MED, lw=0.5, alpha=0.55, zorder=7)
yb = base + 0.28
ax.plot(hx + 0.14 + t * (hw - 0.28), hy + 0.20 + yb * (hh - 0.42),
        color=RED, lw=1.5, zorder=8)
txt(hx + hw / 2, hy + hh - 0.13,
    "honest spread is physical, not noise", fs=7, fw="bold", fc=DARK, z=9)
txt(hx + hw / 2, hy + 0.09,
    "capacity, sensor bias, imbalance", fs=6.5, fc=MED, z=9)

# ================================================== Step 3: router
c3 = p3x + p3w / 2
ty = PY + PH - 1.05
for i, (lab, sub) in enumerate([
        ("(i) shared-current test",
         r"$|\hat{I}_m-\mathrm{med}\,\hat{I}|/1.4826\,\mathrm{MAD}$"),
        ("(ii) report-vs-current test",
         r"CUSUM of $\Delta\hat{s}_m-$ own $\int\!\hat{I}_m dt$")]):
    yy = ty - i * 0.70
    rbox((c3 - p3w * 0.43, yy - 0.26), p3w * 0.86, 0.52, WHITE,
         ec=DEEP_BLUE, lw=1.2, z=6, pad=0.04)
    txt(c3, yy + 0.09, lab, fs=8, fw="bold", fc=DEEP_BLUE, z=7)
    txt(c3, yy - 0.13, sub, fs=6.5, fc=DARK, z=7)

# decision fork
fy = ty - 1.55
txt(c3, fy + 0.10, "inconsistent?", fs=8.5, fw="bold", fc=DARK)
lw_, lh_ = p3w * 0.40, 0.70
arr(c3 - 0.05, fy - 0.05, c3 - lw_ * 0.55, fy - 0.42, color=RED, lw=2.0, ms=14)
arr(c3 + 0.05, fy - 0.05, c3 + lw_ * 0.55, fy - 0.42, color=GREEN, lw=2.0, ms=14)
rbox((c3 - lw_ * 1.05, fy - 0.42 - lh_), lw_, lh_, RED_L, ec=RED, lw=1.3)
txt(c3 - lw_ * 0.55, fy - 0.42 - lh_ / 2 + 0.13, "YES", fs=8, fw="bold", fc=RED)
txt(c3 - lw_ * 0.55, fy - 0.42 - lh_ / 2 - 0.11, "node fault\nquarantine",
    fs=7, fc=DARK)
rbox((c3 + lw_ * 0.05, fy - 0.42 - lh_), lw_, lh_, GREEN_L, ec=GREEN, lw=1.3)
txt(c3 + lw_ * 0.55, fy - 0.42 - lh_ / 2 + 0.13, "NO", fs=8, fw="bold", fc=GREEN)
txt(c3 + lw_ * 0.55, fy - 0.42 - lh_ / 2 - 0.11,
    "battery anomaly\nflag, keep in fusion", fs=7, fc=DARK)

# ================================================== Step 4: consensus
c4 = p4x + p4w / 2
wy = PY + PH - 1.00
txt(c4, wy + 0.28, "trust-weighted fusion", fs=8.5, fw="bold", fc=DEEP_BLUE)
labels = ["trusted", "suspect", "battery", "quarantined"]
colors = [GREEN, AMBER, LIGHT_BLUE, RED]
weights = [0.80, 0.55, 0.10, 0.0]
for i, (lb, co, wv) in enumerate(zip(labels, colors, weights)):
    yy = wy - 0.10 - i * 0.42
    ax.add_patch(Circle((c4 - p4w * 0.32, yy), 0.075, fc=co, ec=co, zorder=6))
    txt(c4 - p4w * 0.32 + 0.17, yy, lb, fs=7.5, ha="left", fc=DARK)
    bar_x = c4 + p4w * 0.06
    if wv > 0:
        ax.add_patch(Rectangle((bar_x, yy - 0.055), p4w * 0.24 * wv, 0.11,
                               fc=co, ec=co, alpha=0.85, zorder=6))
    else:
        ax.plot([bar_x, bar_x + p4w * 0.05], [yy, yy], color=MED, lw=1.0,
                ls=":", zorder=6)
    txt(bar_x + p4w * 0.27, yy, f"w={wv:.2f}", fs=6.5, ha="left", fc=MED)

oy = PY + 0.34
rbox((p4x + p4w * 0.06, oy), p4w * 0.88, 0.64, VERY_LIGHT_BLUE,
     ec=LIGHT_BLUE, lw=1.4)
txt(c4, oy + 0.40, r"consensus $\bar{s}$", fs=9, fw="bold", fc=DEEP_BLUE)
txt(c4, oy + 0.17, "single pack SOC estimate", fs=7, fc=MED)

for xa, xb in [(p1x + p1w - 0.02, p2x + 0.02),
               (p2x + p2w - 0.02, p3x + 0.02),
               (p3x + p3w - 0.02, p4x + 0.02)]:
    arr(xa, PY + PH * 0.42, xb, PY + PH * 0.42, color=DEEP_BLUE, lw=2.6, ms=18)

# ---------------------------------------------------------------- banner
rbox((M, BY), TW, BH, DEEP_BLUE, lw=0, alpha=0.94, pad=0.07)
if RES.exists():
    R = json.loads(RES.read_text(encoding="utf-8"))
    e7 = R["E7_fault_class_matrix"]
    e1 = R["E1p_ablation"]["_false_positive_check"]
    v3 = "EWMA + gap + router (v3)"
    r1 = "EWMA + gap (round-1 detector)"
    wrong_r1 = np.mean([e1[c][r1]["p_target_quarantined"]["mean"] for c in e1]) * 100
    wrong_v3 = np.mean([e1[c][v3]["p_target_quarantined"]["mean"] for c in e1]) * 100
    node = [c for c in e7 if c.startswith("N")]
    det = np.mean([e7[c]["ABR-full"]["p_target_quarantined"]["mean"] for c in node]) * 100
    fix_fp = e7["none"]["Fixed 3 %"]["tpr_steps"]["mean"] * 100
    rmse = np.mean([e7[c]["ABR-full"]["rmse_consensus"]["mean"] for c in e7]) * 100
    stats = [
        (f"{wrong_r1:.0f} → {wrong_v3:.0f} %",
         "battery faults and link glitches\nwrongly quarantined (round 1 → v3)"),
        (f"{det:.0f} %", "true node faults detected\n(8 classes, 5 seeds)"),
        (f"{fix_fp:.0f} %", "steps a fixed 3 % threshold\nexcludes a healthy node"),
        (f"{rmse:.2f} %", "consensus SOC RMSE\nvs true pack SOC"),
    ]
else:
    stats = [("--", "run revision2_experiments.py first")] * 4
sw_ = TW / len(stats)
for i, (big, small) in enumerate(stats):
    cx = M + sw_ * (i + 0.5)
    txt(cx, BY + BH - 0.34, big, fs=17, fw="bold", fc=WHITE)
    txt(cx, BY + 0.28, small, fs=7.6, fc="#d6eaf8")
    if i:
        ax.plot([M + sw_ * i, M + sw_ * i], [BY + 0.12, BY + BH - 0.12],
                color="#5499c7", lw=0.9, zorder=11)

fig.savefig(OUT / "graphical_abstract.pdf", bbox_inches="tight", facecolor=WHITE)
fig.savefig(OUT / "graphical_abstract.png", dpi=300, bbox_inches="tight",
            facecolor=WHITE)
print("wrote", OUT / "graphical_abstract.pdf")
print("wrote", OUT / "graphical_abstract.png")
