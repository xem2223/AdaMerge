"""
Attention Matrix 크기 시각화
- (좌) 실제 attention matrix가 어떻게 생겼는지 예시 heatmap
- (우) 해상도가 커질수록 attention matrix(N x N)의 크기가 quadratic하게
       커지는 것을 정사각형 면적 비교로 시각화

사용법:
    python plot_attention_matrix_size.py
    -> attention_matrix_size.png 저장
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

rng = np.random.default_rng(0)

# ----------------------------------------------------------------------
# (좌) 예시 attention matrix heatmap
# ----------------------------------------------------------------------
N_EXAMPLE = 32  # 시각적으로 보기 좋은 작은 크기로 예시 생성
logits = rng.normal(size=(N_EXAMPLE, N_EXAMPLE))
attn = np.exp(logits) / np.exp(logits).sum(axis=-1, keepdims=True)  # softmax

# ----------------------------------------------------------------------
# (우) 해상도별 attention matrix 크기(N x N) 비교
# 224px -> N=197 (14x14 patch + CLS), 384px -> N=577 (24x24+CLS), 512px -> N=1025 (32x32+CLS)
# ----------------------------------------------------------------------
configs = [
    ("224px", 197),
    ("384px", 577),
    ("512px", 1025),
]

# ----------------------------------------------------------------------
# Plot
# ----------------------------------------------------------------------
plt.rcParams.update({
    "font.size": 12,
    "axes.labelsize": 13,
    "axes.titlesize": 13,
    "legend.fontsize": 10,
})

fig, axes = plt.subplots(1, 2, figsize=(11, 5), dpi=200)

# --- 좌: heatmap ---
ax0 = axes[0]
im = ax0.imshow(attn, cmap="viridis")
ax0.set_title(f"Attention Matrix Example ($N={N_EXAMPLE}$)")
ax0.set_xlabel("Key Tokens")
ax0.set_ylabel("Query Tokens")
ax0.set_xticks([])
ax0.set_yticks([])
cbar = fig.colorbar(im, ax=ax0, fraction=0.046, pad=0.04)
cbar.set_label("Attention Weight")

# --- 우: 정사각형 면적 비교 (N x N 크기를 실제 면적으로 표현) ---
ax1 = axes[1]
max_n = max(n for _, n in configs)
colors = ["#4C72B0", "#DD8452", "#C44E52"]

# 정사각형을 왼쪽 하단 정렬로 겹쳐서 그려 크기 차이를 직관적으로 비교
for (label, n), color in zip(sorted(configs, key=lambda x: -x[1]), colors[::-1]):
    side = n / max_n  # 정규화된 한 변의 길이 (실제 N에 비례 -> 면적은 N^2에 비례)
    rect = patches.Rectangle((0, 0), side, side, linewidth=2,
                              edgecolor=color, facecolor=color, alpha=0.35,
                              label=f"{label} (N={n}, matrix={n}x{n} = {n*n:,} elements)")
    ax1.add_patch(rect)

ax1.set_xlim(0, 1.05)
ax1.set_ylim(0, 1.05)
ax1.set_aspect("equal")
ax1.set_xticks([])
ax1.set_yticks([])
ax1.set_title("Attention Matrix Size Comparison ($N \\times N$)")
ax1.legend(loc="upper left", bbox_to_anchor=(0, -0.05), frameon=False, fontsize=9)

fig.tight_layout()
fig.savefig("attention_matrix_size.png", dpi=300, bbox_inches="tight")
# fig.savefig("attention_matrix_size.pdf", bbox_inches="tight")

print("Saved: attention_matrix_size.png")