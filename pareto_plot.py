import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ── Data ─────────────────────────────────────────────
adamerge = dict(
    label="AdaMerge (ours)",
    flops = np.array([16.10, 15.36, 14.98, 14.39, 13.94, 13.42, 13.22]),
    acc   = np.array([84.76, 84.49, 84.49, 84.28, 84.25, 84.13, 84.11]),
    speed = np.array([ 1.01,  1.06,  1.09,  1.13,  1.17,  1.22,  1.23]),
)

tome = dict(
    label="ToMe",
    flops = np.array([15.93, 15.43, 14.93, 14.43, 13.94, 13.44]),
    acc   = np.array([84.69, 84.44, 84.12, 83.67, 83.11, 82.46]),
    speed = np.array([ 1.10,  1.10,  1.15,  1.19,  1.25,  1.31]),
)

pitome = dict(
    label="PiToMe",
    flops = np.array([15.93, 15.43, 14.93, 14.43, 13.94, 13.44]),
    acc   = np.array([84.57, 84.44, 84.29, 84.22, 83.65, 83.74]),
    speed = np.array([ 1.00,  1.04,  1.08,  1.12,  1.17,  1.23]),
)

others = [tome, pitome]

# ── Pareto envelope helper (speedup용) ───────────────
def pareto_envelope(x, y):
    idx = np.argsort(x)
    x, y = x[idx], y[idx]
    best = []
    max_y = -np.inf
    for xi, yi in zip(x[::-1], y[::-1]):
        if yi > max_y:
            best.append((xi, yi))
            max_y = yi
    best = np.array(best[::-1])
    return best[:, 0], best[:, 1]

# ── Style ────────────────────────────────────────────
PRIMARY = "#1f77b4"

fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))

for ax in axes:
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(False)
    ax.tick_params(labelsize=9)

# ── (a) FLOPs–Accuracy ───────────────────────────────
ax = axes[0]

for d in others:
    ax.scatter(d['flops'], d['acc'],
               color="gray", alpha=0.45, s=22)

ax.plot(adamerge['flops'], adamerge['acc'],
        color=PRIMARY, linewidth=3.0, zorder=5)
ax.scatter(adamerge['flops'], adamerge['acc'],
           color=PRIMARY, s=36, zorder=6)

# FIX: fill_between 직접 사용 (pareto_envelope 불필요)
ax.fill_between(adamerge['flops'], adamerge['acc'], 82,
                color=PRIMARY, alpha=0.08)

ax.set_xlabel("FLOPs (G)", fontsize=10)
ax.set_ylabel("Top-1 Accuracy (%)", fontsize=10)
ax.set_xlim(12.8, 18.0)
ax.set_ylim(82.0, 85.5)
ax.invert_xaxis()
ax.text(15.2, 84.9, "Pareto frontier (ours)", color=PRIMARY, fontsize=9)

# ── (b) Speedup–Accuracy ─────────────────────────────
ax = axes[1]

for d in others:
    ax.scatter(d['speed'], d['acc'],
               color="gray", alpha=0.45, s=22)

ax.plot(adamerge['speed'], adamerge['acc'],
        color=PRIMARY, linewidth=3.0, zorder=5)
ax.scatter(adamerge['speed'], adamerge['acc'],
           color=PRIMARY, s=36, zorder=6)

# speedup은 pareto_envelope 정상 작동
px, py = pareto_envelope(adamerge['speed'], adamerge['acc'])
ax.fill_between(px, py, 82, color=PRIMARY, alpha=0.08)

ax.set_xlabel("Throughput Speedup (×)", fontsize=10)
ax.set_ylabel("Top-1 Accuracy (%)", fontsize=10)
ax.set_xlim(0.95, 1.38)
ax.set_ylim(82.0, 85.5)

# FIX: "Dominates" → "Higher accuracy at any given speedup"
ax.text(1.02, 84.9, "Higher accuracy at any given speedup",
        color=PRIMARY, fontsize=9)

# ── Legend (FIX: 마커 색상 올바르게) ─────────────────
handles = [
    Line2D([0], [0], color=PRIMARY, linewidth=2,
           marker='o', markersize=5, label="AdaMerge (ours)"),
    Line2D([0], [0], color='gray', alpha=0.6, linewidth=0,
           marker='o', markersize=5, label="Other methods"),
]
fig.legend(handles=handles,
           loc='upper center', ncol=2, frameon=False, fontsize=9)

plt.tight_layout()
plt.savefig("cvpr_pareto_final.pdf", bbox_inches='tight')
plt.savefig("cvpr_pareto_final.png", dpi=300, bbox_inches='tight')
print("Done")