"""
adamerge.py
-----------
AdaMerge 핵심 모듈.

[변경 사항]
공간 유사도 제거 → salience 가중 feature similarity만 사용

동작 순서:
  1. Token Salience 계산 (SBAM 방식)
  2. Salience 가중 Feature Similarity 계산
  3. Adaptive r 결정 (유사도 분포 기반)
  4. Bipartite Soft Matching
  5. Salience 가중 평균 병합
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional

from salience import compute_token_salience


# ─────────────────────────────────────────
# 1. Salience 가중 Feature Similarity
# ─────────────────────────────────────────

def compute_salience_weighted_similarity(
    x_A: torch.Tensor,
    x_B: torch.Tensor,
    S: torch.Tensor,
) -> torch.Tensor:
    """
    Salience 가중 Feature Similarity 계산.

    s(i,j) = S_i · cosine(x_A_i, x_B_j)

    salience가 높은 토큰(중요한 토큰)이
    병합 쌍 선택 시 대표로 선택되도록 유도.

    Args:
        x_A: 그룹 A 토큰 [N, T/2, D]
        x_B: 그룹 B 토큰 [N, T/2, D]
        S:   전체 토큰 salience [N, T]

    Returns:
        sim: [N, T/2, T/2]
    """
    N, T_half, D = x_A.shape

    # A 그룹 salience
    S_A = S[:, :T_half]  # [N, T/2]

    # 코사인 유사도
    x_A_norm = F.normalize(x_A, dim=-1)   # [N, T/2, D]
    x_B_norm = F.normalize(x_B, dim=-1)   # [N, T/2, D]
    feat_sim = torch.bmm(x_A_norm, x_B_norm.transpose(1, 2))  # [N, T/2, T/2]

    # Salience 가중
    S_A_expanded = S_A.unsqueeze(-1)       # [N, T/2, 1]
    sim = S_A_expanded * feat_sim          # [N, T/2, T/2]

    return sim


# ─────────────────────────────────────────
# 2. Adaptive r 결정
# ─────────────────────────────────────────

def compute_adaptive_r(
    sim: torch.Tensor,
    mu_l: float,
    sigma_l: float,
    r_max: int,
    alpha: float = 3.0,
) -> int:
    """
    유사도 분포 기반 adaptive r 결정.

    r_l = floor(r_max · sigmoid(α · (S_bar - μ_l) / σ_l))

    유사도가 평균보다 높음 → 토큰들이 유사 → 많이 병합
    유사도가 평균보다 낮음 → 토큰들이 다양 → 적게 병합

    Args:
        sim:     [N, T/2, T/2]
        mu_l:    레이어 l의 평균 유사도 (사전 통계)
        sigma_l: 레이어 l의 유사도 표준편차 (사전 통계)
        r_max:   최대 병합 수
        alpha:   민감도

    Returns:
        r: 병합할 토큰 수 (int)
    """
    # 각 A 토큰의 최대 유사도 → 배치 평균
    max_sim = sim.amax(dim=-1)   # [N, T/2]
    S_bar = max_sim.mean().item()

    # NaN / Inf 방어
    if not math.isfinite(S_bar):
        return r_max // 2  # NaN이면 기본값 반환

    sigma_safe = max(sigma_l, 1e-4)
    S_hat = (S_bar - mu_l) / sigma_safe

    # S_hat도 NaN / Inf 방어
    if not math.isfinite(S_hat):
        return r_max // 2

    sigmoid_val = torch.sigmoid(torch.tensor(alpha * S_hat, dtype=torch.float32)).item()

    # sigmoid_val NaN 방어
    if not math.isfinite(sigmoid_val):
        return r_max // 2

    r = math.floor(r_max * sigmoid_val)
    r = max(0, min(r, r_max))

    return r


# ─────────────────────────────────────────
# 3. Bipartite Matching + Salience 가중 병합 (벡터화)
# ─────────────────────────────────────────

def bipartite_merge(
    x: torch.Tensor,
    S: torch.Tensor,
    r: int,
    sim: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Bipartite Soft Matching + Salience 가중 평균 병합 (완전 벡터화).

    Python for-loop 없이 순수 텐서 연산으로 구현.
    → GPU 병렬 연산 최대 활용.

    Args:
        x:   전체 토큰 [N, T, D]
        S:   salience [N, T]
        r:   병합할 토큰 수
        sim: [N, T/2, T/2]

    Returns:
        x_merged: [N, T-r, D]
        S_merged: [N, T-r]
    """
    N, T, D = x.shape
    T_half = T // 2

    x_A = x[:, :T_half]   # [N, T/2, D]
    x_B = x[:, T_half:]   # [N, T/2, D]
    S_A = S[:, :T_half]   # [N, T/2]
    S_B = S[:, T_half:]   # [N, T/2]

    # r이 T_half를 초과하면 topk 에러 → clamp
    r = min(r, T_half)
    if r == 0:
        return x, S

    # Step 1. 각 A 토큰의 가장 유사한 B 인덱스
    best_match = sim.argmax(dim=-1)        # [N, T/2]

    # Step 2. 유사도 상위 r개 A 토큰 선택
    top_sim = sim.amax(dim=-1)             # [N, T/2]
    _, top_idx = top_sim.topk(r, dim=-1)  # [N, r]

    # Step 3. 병합할 A 토큰 마스크
    merge_mask = torch.zeros(N, T_half, dtype=torch.bool, device=x.device)
    merge_mask.scatter_(1, top_idx, True)  # [N, T/2] True=병합할 A 토큰

    # Step 4. 병합할 A 토큰의 B 대상 인덱스
    b_idx = best_match.gather(1, top_idx)  # [N, r]

    # Step 5. Salience 가중 평균 병합 (벡터화)
    # 병합할 A 토큰들의 feature, salience
    x_A_merge = x_A.gather(                         # [N, r, D]
        1, top_idx.unsqueeze(-1).expand(-1, -1, D)
    )
    S_A_merge = S_A.gather(1, top_idx)              # [N, r]

    # 대응하는 B 토큰들의 feature, salience
    x_B_target = x_B.gather(                        # [N, r, D]
        1, b_idx.unsqueeze(-1).expand(-1, -1, D)
    )
    S_B_target = S_B.gather(1, b_idx)               # [N, r]

    # Salience 가중 평균
    denom = S_A_merge + S_B_target + 1e-6           # [N, r]
    x_merged_vals = (
        S_A_merge.unsqueeze(-1) * x_A_merge +
        S_B_target.unsqueeze(-1) * x_B_target
    ) / denom.unsqueeze(-1)                          # [N, r, D]
    S_merged_vals = torch.max(S_A_merge, S_B_target) # [N, r]

    # Step 6. B 토큰에 병합 결과 scatter
    x_B_out = x_B.clone()
    S_B_out = S_B.clone()

    x_B_out.scatter_(
        1,
        b_idx.unsqueeze(-1).expand(-1, -1, D),
        x_merged_vals
    )
    S_B_out.scatter_(1, b_idx, S_merged_vals)

    # Step 7. 병합되지 않은 A 토큰만 남기기
    keep_mask = ~merge_mask                          # [N, T/2]

    # keep_mask로 A 토큰 필터링 (벡터화)
    # 배치 내 모든 이미지가 동일한 r을 쓰므로 shape 통일
    x_A_keep = x_A[keep_mask].view(N, T_half - r, D)
    S_A_keep = S_A[keep_mask].view(N, T_half - r)

    # 최종 병합
    x_out = torch.cat([x_A_keep, x_B_out], dim=1)  # [N, T-r, D]
    S_out = torch.cat([S_A_keep, S_B_out], dim=1)  # [N, T-r]

    x_merged = x_out
    S_merged = S_out

    return x_merged, S_merged


# ─────────────────────────────────────────
# 4. AdaMerge 메인 클래스
# ─────────────────────────────────────────

class AdaMerge(nn.Module):
    """
    AdaMerge: Adaptive Token Merging via Salience-weighted Similarity

    각 Transformer 레이어에 삽입하여 사용.
    추가 학습 파라미터 없음 (training-free).

    핵심:
      - 병합 쌍 선택: salience 가중 feature similarity
      - 병합 비율:    feature 분포 기반 adaptive r
      - 병합 방식:    salience 가중 평균
    """

    def __init__(
        self,
        num_layers: int = 12,
        r_max: int = 8,
        alpha: float = 3.0,
        stats: Optional[dict] = None,
    ):
        """
        Args:
            num_layers: ViT 레이어 수 (default: 12)
            r_max:      레이어당 최대 병합 수
            alpha:      adaptive r 민감도
            stats:      {layer_idx: {"mu": float, "sigma": float}}
        """
        super().__init__()

        self.num_layers = num_layers
        self.r_max = r_max
        self.alpha = alpha

        if stats is None:
            self.stats = {l: {"mu": 0.5, "sigma": 0.1} for l in range(num_layers)}
        else:
            self.stats = stats

    def forward(
        self,
        x: torch.Tensor,
        layer_idx: int,
    ) -> torch.Tensor:
        """
        Args:
            x:         토큰 시퀀스 [N, T+1, D] (CLS 포함, T=196)
            layer_idx: 현재 레이어 인덱스 (0~11)

        Returns:
            x_out: 병합 후 토큰 [N, T'+1, D]
        """
        # CLS 토큰 분리
        cls_token = x[:, :1, :]      # [N, 1, D]
        patch_tokens = x[:, 1:, :]   # [N, T, D]
        T = patch_tokens.shape[1]
        T_half = T // 2

        # Step 1. Salience 계산
        S = compute_token_salience(patch_tokens)  # [N, T]

        # Step 2. 토큰 분할
        x_A = patch_tokens[:, :T_half]
        x_B = patch_tokens[:, T_half:]

        # Step 3. Salience 가중 feature similarity
        sim = compute_salience_weighted_similarity(x_A, x_B, S)  # [N, T/2, T/2]

        # Step 4. Adaptive r 결정
        mu_l    = self.stats[layer_idx]["mu"]
        sigma_l = self.stats[layer_idx]["sigma"]
        r = compute_adaptive_r(sim, mu_l, sigma_l, self.r_max, self.alpha)

        # Step 5. Salience 가중 병합
        patch_merged, _ = bipartite_merge(patch_tokens, S, r, sim)

        # CLS 재결합
        x_out = torch.cat([cls_token, patch_merged], dim=1)

        return x_out


if __name__ == "__main__":
    N, T, D = 2, 197, 768

    x = torch.randn(N, T, D)

    # 실제 통계값으로 테스트
    test_stats = {
        0:  {"mu": 0.1736, "sigma": 0.1105},
        1:  {"mu": 0.1676, "sigma": 0.0888},
        2:  {"mu": 0.1334, "sigma": 0.0800},
        3:  {"mu": 0.1464, "sigma": 0.0809},
        4:  {"mu": 0.1682, "sigma": 0.0779},
        5:  {"mu": 0.1727, "sigma": 0.0787},
        6:  {"mu": 0.1630, "sigma": 0.0783},
        7:  {"mu": 0.1444, "sigma": 0.0831},
        8:  {"mu": 0.1086, "sigma": 0.0790},
        9:  {"mu": 0.0805, "sigma": 0.0618},
        10: {"mu": 0.0260, "sigma": 0.0198},
        11: {"mu": 0.0164, "sigma": 0.0166},
    }

    model = AdaMerge(
        num_layers=12,
        r_max=8,
        alpha=3.0,
        stats=test_stats,
    )

    print("=== AdaMerge 동작 확인 ===")
    print(f"입력 shape: {x.shape}")

    for l in range(12):
        x_prev = x
        x = model(x, layer_idx=l)
        merged = x_prev.shape[1] - x.shape[1]
        print(f"레이어 {l:2d} 후 → shape: {x.shape}, 병합={merged}개")

    print(f"\n최종 출력 shape: {x.shape}")
    print(f"총 병합된 토큰 수: {197 - x.shape[1]}")
    print("완료!")