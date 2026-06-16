#!/usr/bin/env python3
"""
Generate publication-quality graphical abstract for:
  ABR-PINN: Physics-Informed Byzantine-Resilient Consensus
  for Distributed Battery Management Systems

Target: Elsevier Applied Energy
Requirements: landscape, min 530x1200 px, vector PDF preferred.
Output: ../graphical_abstract.pdf
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle
import numpy as np
from pathlib import Path

# ──────────────────────────────────────────────
# Color palette
# ──────────────────────────────────────────────
DEEP_BLUE       = "#1a5276"
LIGHT_BLUE      = "#2980b9"
VERY_LIGHT_BLUE = "#d6eaf8"
GREEN           = "#27ae60"
GREEN_DARK      = "#1e8449"
AMBER           = "#f39c12"
RED             = "#e74c3c"
WHITE           = "#ffffff"
OFF_WHITE       = "#fdfefe"
DARK_GRAY       = "#2c3e50"
MED_GRAY        = "#7f8c8d"
PANEL_BG        = "#f8f9fa"

# ──────────────────────────────────────────────
# Figure
# ──────────────────────────────────────────────
FW, FH = 14.0, 6.0
fig, ax = plt.subplots(figsize=(FW, FH), dpi=300)
ax.set_xlim(0, FW)
ax.set_ylim(0, FH)
ax.set_aspect("equal")
ax.axis("off")
fig.patch.set_facecolor(WHITE)

# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────
def rbox(xy, w, h, fc, ec=None, lw=1.5, alpha=1.0, zorder=2, pad=0.06):
    ec = ec or fc
    ax.add_patch(FancyBboxPatch(xy, w, h, boxstyle=f"round,pad={pad}",
                                fc=fc, ec=ec, lw=lw, alpha=alpha, zorder=zorder))

def circ(cx, cy, r, fc, ec=None, lw=1.5, zorder=4):
    ax.add_patch(Circle((cx, cy), r, fc=fc, ec=(ec or fc), lw=lw, zorder=zorder))

def txt(x, y, s, fs=8, fw="normal", fc=DARK_GRAY, ha="center", va="center",
        zorder=10, **kw):
    ax.text(x, y, s, fontsize=fs, fontweight=fw, color=fc, ha=ha, va=va,
            fontfamily="serif", zorder=zorder, **kw)

def arr(x0, y0, x1, y1, color=DEEP_BLUE, lw=3.5, ms=22):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                                mutation_scale=ms),
                zorder=12)

def sarr(x0, y0, x1, y1, color=DEEP_BLUE, lw=1.5):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                                mutation_scale=12),
                zorder=5)

# ──────────────────────────────────────────────
# Layout constants
# ──────────────────────────────────────────────
M = 0.20  # margin
TY, TH = 5.30, 0.55   # title bar
BY, BH = 0.08, 1.05   # bottom banner
PY = BY + BH + 0.15   # panel y
PH = TY - PY - 0.12   # panel height
TW = FW - 2*M         # total width

# Panel proportions
p1w = TW * 0.30;  p1x = M
p2w = TW * 0.36;  p2x = p1x + p1w
p3w = TW * 0.34;  p3x = p2x + p2w

# ──────────────────────────────────────────────
# TITLE BAR
# ──────────────────────────────────────────────
rbox((M, TY), TW, TH, DEEP_BLUE, lw=0, pad=0.08)
txt(FW/2, TY + TH/2,
    "ABR-PINN: Physics-Informed Byzantine-Resilient Consensus for Distributed BMS",
    fs=14, fw="bold", fc=WHITE)

# ──────────────────────────────────────────────
# PANEL BACKGROUNDS + HEADERS
# ──────────────────────────────────────────────
for px, pw, ph, sub in [
    (p1x, p1w, "Phase 1", "Local SOC Estimation"),
    (p2x, p2w, "Phase 2", "Statistical Byzantine Detection"),
    (p3x, p3w, "Phase 3", "Trust-Weighted Consensus"),
]:
    rbox((px, PY), pw, PH, PANEL_BG, ec="#d5d8dc", lw=1.0, alpha=0.6)
    hh = 0.55
    rbox((px, PY + PH - hh), pw, hh, DEEP_BLUE, lw=0, alpha=0.10, pad=0.0)
    txt(px + pw/2, PY + PH - 0.17, ph, fs=12, fw="bold", fc=DEEP_BLUE)
    txt(px + pw/2, PY + PH - 0.42, sub, fs=8, fc=MED_GRAY)

# ==============================================================
# PHASE 1 — Battery modules (top) + PINN (bottom)
# ==============================================================
p1cx = p1x + p1w/2

# Battery grid: 4x4 compact
bw, bh = 0.30, 0.36
bgx, bgy = 0.40, 0.46
g_y0 = PY + PH - 0.95  # top of battery grid

for r in range(4):
    for c in range(4):
        cx = p1cx - 1.5 * bgx + c * bgx
        cy = g_y0 - r * bgy
        n = r * 4 + c + 1
        rbox((cx - bw/2, cy - bh/2), bw, bh,
             LIGHT_BLUE, ec=DEEP_BLUE, lw=0.7, alpha=0.85, zorder=3, pad=0.03)
        # terminal
        nw = bw * 0.28
        rbox((cx - nw/2, cy + bh/2 - 0.004), nw, 0.03,
             DARK_GRAY, lw=0, zorder=4, pad=0.01)
        txt(cx, cy - 0.01, str(n), fs=5.5, fw="bold", fc=WHITE, zorder=5)

# "16 modules" label
txt(p1cx, g_y0 + bh/2 + 0.12, "16 LiFePO₄ modules", fs=7, fc=MED_GRAY)

# PINN box below grid
pw_box = p1w * 0.82
ph_box = 1.05
px_box = p1x + (p1w - pw_box) / 2
py_box = PY + 0.20
rbox((px_box, py_box), pw_box, ph_box,
     VERY_LIGHT_BLUE, ec=LIGHT_BLUE, lw=1.5, alpha=0.95)

nn_cx = px_box + pw_box / 2
nn_cy = py_box + ph_box / 2 + 0.12

# Neural network diagram
layers = [3, 5, 2]
lsp = 0.38
nr = 0.048
nn_x0 = nn_cx - (len(layers)-1) * lsp / 2
nn_p = {}
for li, n in enumerate(layers):
    lx = nn_x0 + li * lsp
    th = (n-1) * 0.14
    for ni in range(n):
        ny = nn_cy - th/2 + ni * 0.14
        nn_p[(li, ni)] = (lx, ny)

for li in range(len(layers)-1):
    for ni in range(layers[li]):
        for nj in range(layers[li+1]):
            x1, y1 = nn_p[(li, ni)]; x2, y2 = nn_p[(li+1, nj)]
            ax.plot([x1, x2], [y1, y2], color=LIGHT_BLUE, lw=0.35, alpha=0.5, zorder=4)
for p in nn_p.values():
    circ(*p, nr, DEEP_BLUE, lw=0.5, zorder=5)

txt(nn_cx, nn_cy - 0.45, "PINN", fs=10, fw="bold", fc=DEEP_BLUE)
txt(nn_cx, py_box + 0.10,
    r"$\int i\,dt$ + $\mathcal{L}_{\mathrm{phys}}$", fs=7.5, fc=DARK_GRAY)

# Arrow grid -> PINN
grid_bot = g_y0 - 3 * bgy - bh/2
arr(p1cx, grid_bot - 0.04, p1cx, py_box + ph_box + 0.04,
    color=MED_GRAY, lw=2.0, ms=16)

# Output label
txt(p1cx, py_box + ph_box + 0.42,
    r"$\widehat{\mathrm{SOC}}_i$  ($i\!=\!1\ldots16$)",
    fs=7.5, fw="bold", fc=DEEP_BLUE)

# ==============================================================
# PHASE 2 — Detection pipeline + trust classification
# ==============================================================
p2cx = p2x + p2w/2

# Three processing steps
sw, sh = 2.7, 0.38
slabels = ["MAD z-score Computation",
           "Asymmetric EWMA Filtering",
           "Gap-based Quarantine"]
sy0 = PY + PH - 1.00
sdy = 0.62

for i, lab in enumerate(slabels):
    sy = sy0 - i * sdy
    rbox((p2cx - sw/2, sy - sh/2), sw, sh,
         WHITE, ec=DEEP_BLUE, lw=1.3, zorder=6, pad=0.04)
    txt(p2cx, sy, lab, fs=8, fw="bold", fc=DEEP_BLUE, zorder=7)
    if i < len(slabels) - 1:
        sarr(p2cx, sy - sh/2 - 0.04,
             p2cx, sy - sdy + sh/2 + 0.04,
             color=DEEP_BLUE, lw=1.8)

# Mini z-score chart
mx = p2x + p2w - 1.55
my_top = sy0 - sdy + 0.02
mw, mh = 1.30, 0.70
rbox((mx, my_top - mh), mw, mh, WHITE, ec="#bdc3c7", lw=0.8, zorder=5, pad=0.02)

th_y = my_top - mh * 0.30
ax.plot([mx + 0.08, mx + mw - 0.08], [th_y, th_y],
        color=RED, lw=1.2, ls="--", zorder=7)
txt(mx + mw - 0.10, th_y + 0.06, r"$\tau$", fs=7, fc=RED, ha="right", zorder=8)

np.random.seed(42)
xs_p = np.linspace(mx + 0.10, mx + mw - 0.10, 12)
ys_ok = (my_top - mh + 0.06) + np.random.uniform(0, mh * 0.42, 9)
ys_bad = th_y + np.random.uniform(0.05, mh * 0.22, 3)
for j in range(9):
    ax.plot(xs_p[j], ys_ok[j], "o", color=GREEN, ms=3.0, zorder=8)
for j in range(3):
    ax.plot(xs_p[9+j], ys_bad[j], "o", color=RED, ms=3.0, zorder=8)
txt(mx + mw/2, my_top - mh - 0.08, "EWMA z-score", fs=5.5, fc=MED_GRAY)

# Arrow: quarantine box -> trust classification
quarantine_y = sy0 - 2 * sdy
sarr(p2cx, quarantine_y - sh/2 - 0.04,
     p2cx, PY + 1.15 + 0.04,
     color=DEEP_BLUE, lw=1.8)

# Trust node classification
tr = 0.135
nsp = 2 * tr + 0.08
t_base = PY + 0.92
lbl_x = p2x + 0.18

# Trusted: 2 rows of 5
for sr in range(2):
    ry = t_base + (1 - sr) * 0.38
    nx0 = p2cx - 5 * nsp / 2 + tr + 0.40
    if sr == 0:
        txt(lbl_x, ry, "Trusted", fs=7, fw="bold", fc=GREEN, ha="left")
    for j in range(5):
        nx = nx0 + j * nsp
        m = sr * 5 + j + 1
        circ(nx, ry, tr, GREEN, lw=1.0, zorder=5)
        txt(nx, ry, str(m), fs=5.5, fw="bold", fc=WHITE, zorder=7)

# Suspect
ry_s = t_base - 0.33
nx0_s = p2cx - 3 * nsp / 2 + tr + 0.40
txt(lbl_x, ry_s, "Suspect", fs=7, fw="bold", fc=AMBER, ha="left")
for j in range(3):
    nx = nx0_s + j * nsp
    circ(nx, ry_s, tr, AMBER, lw=1.0, zorder=5)
    txt(nx, ry_s, str(11+j), fs=5.5, fw="bold", fc=WHITE, zorder=7)

# Byzantine (with X)
ry_b = t_base - 0.66
nx0_b = p2cx - 3 * nsp / 2 + tr + 0.40
txt(lbl_x, ry_b, "Byzantine", fs=7, fw="bold", fc=RED, ha="left")
for j in range(3):
    nx = nx0_b + j * nsp
    circ(nx, ry_b, tr, RED, lw=1.0, zorder=5)
    txt(nx, ry_b, str(14+j), fs=5.5, fw="bold", fc=WHITE, zorder=7)
    sz = 0.075
    ax.plot([nx-sz, nx+sz], [ry_b-sz, ry_b+sz], color=WHITE, lw=2, zorder=8)
    ax.plot([nx-sz, nx+sz], [ry_b+sz, ry_b-sz], color=WHITE, lw=2, zorder=8)

# ==============================================================
# PHASE 3 — Trust-Weighted Consensus
# ==============================================================
p3cx = p3x + p3w / 2

# Centering: sigma at panel center
scx = p3cx - 0.05
scy = PY + PH/2 + 0.50

# Input labels + arrows
inputs = [
    (r"$w_1 \!\cdot\! \widehat{\mathrm{SOC}}_1$", GREEN,   0.90),
    (r"$w_2 \!\cdot\! \widehat{\mathrm{SOC}}_2$", GREEN,   0.50),
    (None,                                         MED_GRAY, 0.10),
    (r"$w_k \!\cdot\! \widehat{\mathrm{SOC}}_k$", AMBER,  -0.30),
    (r"$w_{\mathrm{byz}} \!=\! 0$",               RED,    -0.70),
]

ix0 = p3x + 0.12
afrom = ix0 + 1.35

for lab, col, dy in inputs:
    iy = scy + dy
    if lab is None:
        txt(ix0 + 0.50, iy, r"$\vdots$", fs=12, fc=MED_GRAY)
        continue
    txt(ix0 + 0.06, iy, lab, fs=7, fc=col, ha="left")
    sarr(afrom, iy, scx - 0.31, scy, color=col, lw=1.0)

# Sigma
circ(scx, scy, 0.28, DEEP_BLUE, lw=2.0, zorder=6)
txt(scx, scy, r"$\Sigma$", fs=16, fw="bold", fc=WHITE, zorder=8)

# Output arrow + box
obx = scx + 0.90
oby = scy - 0.42
obw, obh = 1.40, 0.84

arr(scx + 0.31, scy, obx + 0.03, scy, color=GREEN_DARK, lw=3.0, ms=20)

rbox((obx, oby), obw, obh, GREEN, ec=GREEN_DARK, lw=2.0, zorder=6, pad=0.06)
txt(obx + obw/2, scy + 0.14, "Consensus", fs=9.5, fw="bold", fc=WHITE, zorder=8)
txt(obx + obw/2, scy - 0.16, "SOC", fs=13, fw="bold", fc=WHITE, zorder=8)

# Formula
fy = PY + 0.72
txt(p3cx + 0.30, fy,
    r"$\widehat{\mathrm{SOC}}_{\mathrm{pack}} = "
    r"\frac{\sum_{i \in \mathcal{T}} w_i \,"
    r"\widehat{\mathrm{SOC}}_i}"
    r"{\sum_{i \in \mathcal{T}} w_i}$",
    fs=10, fc=DARK_GRAY)
txt(p3cx + 0.30, fy - 0.30,
    r"$\mathcal{T}$: trusted set,   $w_i \propto$ trust score",
    fs=6.5, fc=MED_GRAY)

# ==============================================================
# INTER-PANEL ARROWS
# ==============================================================
ay = PY + PH/2 + 0.35
arr(p1x + p1w - 0.05, ay, p2x + 0.08, ay)
arr(p2x + p2w - 0.05, ay, p3x + 0.08, ay)

# ==============================================================
# BOTTOM BANNER
# ==============================================================
rbox((M, BY), TW, BH, DEEP_BLUE, lw=0, pad=0.08)

results = [
    ("100%",               "False-Positive Reduction",
     "FPR: 0.00% vs 5.13%"),
    ("94.2%",              "Mean True Positive Rate",
     "Byzantine detection accuracy"),
    ("16-Module LiFePO₄", "Solar Testbed",
     "70 days  |  22,524 samples"),
]

bw = TW / len(results)
for i, (big, sub, det) in enumerate(results):
    bx = M + i * bw + bw / 2
    by = BY + BH / 2
    txt(bx, by + 0.28, big, fs=18, fw="bold", fc=WHITE)
    txt(bx, by - 0.05, sub, fs=8.5, fw="bold", fc=AMBER, linespacing=1.1)
    txt(bx, by - 0.32, det, fs=6.5, fc="#bdc3c7")
    if i < len(results) - 1:
        sx = M + (i+1) * bw
        ax.plot([sx, sx], [BY + 0.12, BY + BH - 0.12],
                color=LIGHT_BLUE, lw=0.8, alpha=0.4, zorder=10)

# ==============================================================
# Save
# ==============================================================
out_dir = Path(__file__).resolve().parent.parent
pdf = out_dir / "graphical_abstract.pdf"
png = out_dir / "graphical_abstract.png"

fig.savefig(str(pdf), format="pdf", bbox_inches="tight",
            pad_inches=0.03, dpi=300, facecolor=WHITE)
fig.savefig(str(png), format="png", bbox_inches="tight",
            pad_inches=0.03, dpi=300, facecolor=WHITE)

print(f"Graphical abstract saved:")
print(f"  PDF: {pdf}")
print(f"  PNG: {png}")
print(f"  Size: {FW}x{FH} in @ 300 DPI = {int(FW*300)}x{int(FH*300)} px")
plt.close()
