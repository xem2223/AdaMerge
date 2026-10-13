"""
vit_large_adamerge.py
---------------------
ViT-Large-patch16에 AdaMerge를 통합한 모델.
vit_adamerge.py (ViT-Base)와 동일한 구조, Large 아키텍처 대응.

ViT-Large 스펙:
  - embed_dim : 1024  (Base: 768,  Small: 384)
  - num_heads : 16    (Base: 12,   Small: 6)
  - num_layers: 24    (Base: 12,   Small: 12)  ← 주의
  - patch_size: 16
  - input_size: 224x224  → 196 tokens + 1 CLS = 197

주의: num_layers=24로 Base/Small과 다름.
      precompute_stats.py 실행 시 --num_layers 24 지정 필요.
"""

import torch
import torch.nn as nn
import timm
from typing import Optional

from adamerge import AdaMerge
from stats_utils import load_stats


class ViTLargeWithAdaMerge(nn.Module):
    def __init__(
        self,
        r_max: int = 8,
        alpha: float = 3.0,
        stats_path: Optional[str] = None,
        pretrained: bool = True,
    ):
        super().__init__()

        # .npz 체크포인트는 timm이 내부적으로 처리
        self.vit = timm.create_model(
            "vit_large_patch16_224",
            pretrained=pretrained,
        )

        if stats_path is not None:
            stats = load_stats(stats_path)
        else:
            print("Warning: stats_path 없음. 기본값 사용.")
            stats = None

        # num_layers=24 : ViT-Large는 Base/Small의 2배
        self.ada_merge = AdaMerge(
            num_layers=24,
            r_max=r_max,
            alpha=alpha,
            stats=stats,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Patch embedding
        x = self.vit.patch_embed(x)
        cls_token = self.vit.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls_token, x], dim=1)
        x = self.vit.pos_drop(x + self.vit.pos_embed)

        # Transformer blocks + AdaMerge (block 전에 merge)
        for l, block in enumerate(self.vit.blocks):
            x = self.ada_merge(x, layer_idx=l)
            x = block(x)

        x = self.vit.norm(x)
        x = x[:, 0]
        x = self.vit.head(x)
        return x


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ViTLargeWithAdaMerge(r_max=8, alpha=3.0, pretrained=False).to(device)
    x = torch.randn(2, 3, 224, 224).to(device)

    with torch.no_grad():
        out = model(x)

    print(f"입력: {x.shape} → 출력: {out.shape}")
    print("vit_large_adamerge.py 완료!")