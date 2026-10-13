"""
salience.py
-----------
SBAM 방식의 Token Salience 계산 모듈.

논문 수식:
  A     = X · X^T               (affinity matrix)
  A_hat = softmax(A)             (row-wise normalization)
  S     = N(sum_j A_hat[:,j,:])  (column-wise sum → normalize)

  각 토큰이 다른 토큰들에게 얼마나 영향을 주는가
  → salience 높음: 중요한 토큰 (병합 시 대표로 살아남음)
  → salience 낮음: 중복 토큰 (병합되어 흡수됨)
"""

import torch
import torch.nn.functional as F


def compute_token_salience(x: torch.Tensor) -> torch.Tensor:
    """
    Token Salience 계산 (SBAM 방식)

    Args:
        x: 토큰 시퀀스 [N, T, D]
           N = 배치 크기
           T = 토큰 수 (196)
           D = 토큰 차원 (768)

    Returns:
        S: salience score [N, T], 값 범위 0~1
    """
    N, T, D = x.shape

    # Step 1. Affinity matrix 계산
    # A[n, i, j] = x[n,i] · x[n,j]
    # shape: [N, T, T]
    A = torch.bmm(x, x.transpose(1, 2))  # [N, T, T]

    # NaN / Inf 방어: 학습 초반 불안정한 feature 대응
    A = torch.nan_to_num(A, nan=0.0, posinf=1.0, neginf=-1.0)

    # Step 2. Row-wise softmax 정규화
    A_hat = F.softmax(A, dim=-1)  # [N, T, T]

    # softmax 결과 NaN 방어 (입력이 모두 -inf인 경우 등)
    A_hat = torch.nan_to_num(A_hat, nan=1.0 / A.shape[-1])

    # Step 3. Column-wise sum → outgoing weight
    outgoing = A_hat.sum(dim=1)  # [N, T]

    # Step 4. Min-max 정규화 → 0~1 범위
    S = _minmax_normalize(outgoing)  # [N, T]

    return S


def _minmax_normalize(x: torch.Tensor) -> torch.Tensor:
    """
    Min-max 정규화: N(x) = (x - min) / (max - min)

    모든 값이 동일한 경우(분산=0) → 균등 salience 0.5 반환

    Args:
        x: [N, T]

    Returns:
        정규화된 텐서 [N, T]
    """
    x_min = x.amin(dim=-1, keepdim=True)  # [N, 1]
    x_max = x.amax(dim=-1, keepdim=True)  # [N, 1]
    denom = x_max - x_min                  # [N, 1]

    # 분산이 거의 없는 경우 (모든 토큰이 동등) → 0.5로 균등 설정
    uniform = torch.full_like(x, 0.5)
    normalized = (x - x_min) / denom.clamp(min=1e-6)

    # denom이 너무 작으면 균등값 사용
    mask = (denom < 1e-6).expand_as(x)
    return torch.where(mask, uniform, normalized)


if __name__ == "__main__":
    # 동작 확인
    N, T, D = 4, 196, 768
    x = torch.randn(N, T, D)

    S = compute_token_salience(x)

    print(f"입력 shape:    {x.shape}")
    print(f"Salience shape: {S.shape}")
    print(f"Salience 범위:  min={S.min():.4f}, max={S.max():.4f}")
    print(f"Salience 평균:  {S.mean():.4f}")
    print("salience.py 동작 확인 완료!")