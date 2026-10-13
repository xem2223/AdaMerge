"""
vit_small_adamerge.py
---------------------
ViT-Small-patch16에 AdaMerge를 통합한 모델.
vit_adamerge.py (ViT-Base)와 동일한 구조, Small 아키텍처 대응.

ViT-Small 스펙:
  - embed_dim : 384  (Base: 768)
  - num_heads : 6    (Base: 12)
  - num_layers: 12   (Base: 12, 동일)
  - patch_size: 16
  - input_size: 224x224  → 196 tokens + 1 CLS = 197
"""

import torch
import torch.nn as nn
import timm
from typing import Optional

from adamerge import AdaMerge
from stats_utils import load_stats


class ViTSmallWithAdaMerge(nn.Module):
    def __init__(
        self,
        r_max: int = 8,
        alpha: float = 3.0,
        stats_path: Optional[str] = None,
        pretrained: bool = True,
    ):
        super().__init__()

        if pretrained:
            import os
            from safetensors.torch import load_file
            self.vit = timm.create_model("vit_small_patch16_224", pretrained=False)
            # TODO: Small 체크포인트 경로로 교체 필요
            cache_path = os.path.expanduser(
                "~/.cache/huggingface/hub/"
                "models--timm--vit_small_patch16_224.augreg_in21k_ft_in1k/"
                "snapshots/7e2c55630205e1266030f18370f4c6ed1a514b52/model.safetensors"
            )
            state_dict = load_file(cache_path)
            self.vit.load_state_dict(state_dict, strict=False)
        else:
            self.vit = timm.create_model("vit_small_patch16_224", pretrained=False)

        if stats_path is not None:
            stats = load_stats(stats_path)
        else:
            print("Warning: stats_path 없음. 기본값 사용.")
            stats = None

        # num_layers=12 : ViT-Small도 Base와 동일
        self.ada_merge = AdaMerge(
            num_layers=12,
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

    # pretrained=False: 체크포인트 경로 세팅 전 shape 확인용
    model = ViTSmallWithAdaMerge(r_max=8, alpha=3.0, pretrained=False).to(device)
    x = torch.randn(2, 3, 224, 224).to(device)

    with torch.no_grad():
        out = model(x)

    print(f"입력: {x.shape} → 출력: {out.shape}")
    print("vit_small_adamerge.py 완료!")