"""
ViT 병목(bottleneck) motivation figure
- Self-attention은 토큰 수 N에 대해 O(N^2), MLP/projection은 O(N)으로 스케일링됨을
  보여줘서 "토큰 수가 늘어날수록 attention이 연산량을 지배한다"는 점을 시각화.
- 논문 Introduction/Motivation 섹션에 넣기 좋은 형태.

사용법:
    python plot_vit_bottleneck.py
    -> vit_bottleneck.png 저장
"""

import numpy as np
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# ViT-Base/16 스펙 (필요시 수정)
# ----------------------------------------------------------------------
D = 768          # embedding dim
L = 12           # transformer layers
MLP_RATIO = 4    # MLP hidden dim = D * MLP_RATIO

# 토큰 수 범위: ViT-B/16 기준 224px -> 197 tokens (CLS 포함)
# 해상도를 키운다고 가정하고 N을 넓게 스캔 (motivation을 위해)
N = np.linspace(50, 1600, 300)

def layer_flops(N, D=D, mlp_ratio=MLP_RATIO):
    """단일 transformer layer의 FLOPs를 attention/MLP로 분해 (approx, multiply-add=1 FLOP 기준)"""
    qkv_proj   = 3 * N * D * D          # Q,K,V projection
    out_proj   = N * D * D              # output projection
    attn_score = N * N * D              # QK^T
    attn_value = N * N * D              # softmax(QK^T) V
    mlp        = 2 * N * D * (mlp_ratio * D)  # fc1 + fc2

    attn_total = qkv_proj + out_proj + attn_score + attn_value
    return attn_total, mlp

attn_flops, mlp_flops = layer_flops(N)
attn_flops *= L
mlp_flops  *= L
total_flops = attn_flops + mlp_flops

# GFLOPs 단위로 변환 (곱셈+덧셈 2 FLOPs로 보정)
to_g = lambda x: x * 2 / 1e9
attn_g, mlp_g, total_g = to_g(attn_flops), to_g(mlp_flops), to_g(total_flops)

# ----------------------------------------------------------------------
# Plot
# ----------------------------------------------------------------------
plt.rcParams.update({
    "font.size": 12,
    "axes.labelsize": 13,
    "axes.titlesize": 14,
    "legend.fontsize": 11,
})

fig, ax = plt.subplots(figsize=(6, 4.5), dpi=200)

ax.plot(N, total_g, color="#C44E52", linewidth=2.5, label="Total FLOPs")
ax.plot(N, attn_g,  color="#4C72B0", linewidth=2.0, linestyle="--", label="Self-Attention FLOPs ($O(N^2)$)")
ax.plot(N, mlp_g,   color="#888888", linewidth=2.0, linestyle="--", label="MLP/Projection FLOPs ($O(N)$)")

# 기준선: ViT-B/16, 224px, N=197
n_ref = 197
attn_ref, mlp_ref = layer_flops(np.array([n_ref]))
ref_total_g = to_g((attn_ref + mlp_ref) * L)[0]
ax.axvline(n_ref, color="black", linewidth=0.8, linestyle=":")
ax.annotate(f"224px\n(N={n_ref})", xy=(n_ref, ref_total_g), xytext=(n_ref + 60, ref_total_g * 0.4),
            fontsize=10, arrowprops=dict(arrowstyle="->", lw=0.8))

ax.set_xlabel("Number of Tokens (N)")
ax.set_ylabel("FLOPs (G)")
ax.set_title("ViT Computational Bottleneck: Attention Scales Quadratically")
ax.grid(True, linestyle="--", alpha=0.4)
ax.legend(loc="upper left", frameon=True)

fig.tight_layout()
fig.savefig("vit_bottleneck.png", dpi=300, bbox_inches="tight")
# fig.savefig("vit_bottleneck.pdf", bbox_inches="tight")  # 논문용 벡터 포맷

print("Saved: vit_bottleneck.png")