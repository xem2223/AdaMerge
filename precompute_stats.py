"""
precompute_stats.py
-------------------
AdaMerge 적용 환경에서 사전 통계 추출.

[수정 이유]
기존: pretrained ViT (196개 토큰) 환경에서 통계 추출
      → 실제 실행 시 레이어 4~11은 이미 172개로 줄어든 상태
      → 통계와 실제 환경 불일치 → 깊은 레이어에서 r ≈ 0

변경: ViTWithAdaMerge를 사용해 실제 실행 환경에서 통계 추출
      → 각 레이어의 실제 토큰 수 기준으로 μ, σ 계산
      → S_hat이 올바르게 계산됨 → 깊은 레이어도 적절히 병합

사용법:
  # 1차: 기본 stats로 초기 실행 (없으면 기본값 사용)
  python precompute_stats.py \
    --data_dir ./data \
    --output   ./stats.json \
    --subset   0.01

  # 2차: 생성된 stats로 다시 추출 (수렴할 때까지 반복 가능)
  python precompute_stats.py \
    --data_dir ./data \
    --output   ./stats.json \
    --prev_stats ./stats.json \
    --subset   0.01
"""

import os
import json
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from salience import compute_token_salience
from adamerge import compute_salience_weighted_similarity, compute_adaptive_r
#ViT-Base stats 추출
#from vit_adamerge import ViTWithAdaMerge
#ViT-Small stats 추출
#from vit_small_adamerge import ViTSmallWithAdaMerge as ViTWithAdaMerge
#ViT-Large stats 추출
from vit_large_adamerge import ViTLargeWithAdaMerge as ViTWithAdaMerge

def compute_feature_similarity_stats(patch: torch.Tensor) -> float:
    """레이어의 현재 patch feature에서 salience 가중 유사도 계산"""
    N, T, D = patch.shape
    T_half = T // 2

    x_A = patch[:, :T_half]
    x_B = patch[:, T_half:]
    S = compute_token_salience(patch)

    sim = compute_salience_weighted_similarity(x_A, x_B, S)
    max_sim = sim.amax(dim=-1)
    return max_sim.mean().item()


def precompute_stats(
    data_dir: str,
    output_path: str,
    prev_stats_path: str = None,
    subset_ratio: float = 0.01,
    num_layers: int = 24,
    r_max: int = 8,
    alpha: float = 3.0,
    batch_size: int = 32,
    device: str = "cuda",
):
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"r_max={r_max}, alpha={alpha}")

    # ── 데이터 로더 ──────────────────────────
    val_transform = transforms.Compose([
        transforms.Resize(249, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
        ),
    ])

    from datasets import load_dataset

    class HFSubset(torch.utils.data.Dataset):
        def __init__(self, hf_dataset, indices, transform=None):
            self.dataset = hf_dataset
            self.indices = indices
            self.transform = transform
        def __len__(self):
            return len(self.indices)
        def __getitem__(self, idx):
            item = self.dataset[int(self.indices[idx])]
            image = item["image"].convert("RGB")
            label = item["label"]
            if self.transform:
                image = self.transform(image)
            return image, label

    print("ImageNet-1k 로드 중...")
    hf_data = load_dataset("ILSVRC/imagenet-1k", cache_dir=data_dir)
    train_hf = hf_data["train"]
    n_total = len(train_hf)
    n_subset = int(n_total * subset_ratio)
    indices = torch.randperm(n_total)[:n_subset].tolist()

    subset = HFSubset(train_hf, indices, transform=val_transform)
    loader = DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    print(f"통계 추출 데이터: {n_subset}장 ({subset_ratio*100:.1f}%)\n")

    # ── AdaMerge 모델 로드 ────────────────────
    # prev_stats가 있으면 사용, 없으면 기본값
    print(f"AdaMerge 모델 로드 중...")
    if prev_stats_path and os.path.exists(prev_stats_path):
        print(f"  이전 stats 사용: {prev_stats_path}")
        model = ViTWithAdaMerge(
            r_max=r_max,
            alpha=alpha,
            stats_path=prev_stats_path,
            pretrained=True,
        )
    else:
        print("  초기 stats 없음 → 기본값(mu=0.5, sigma=0.1) 사용")
        model = ViTWithAdaMerge(
            r_max=r_max,
            alpha=alpha,
            stats_path=None,
            pretrained=True,
        )

    model = model.to(device)
    model.eval()

    # ── 각 레이어에서 실제 토큰 feature 추출 ──
    stats_collector = {l: [] for l in range(num_layers)}

    with torch.no_grad():
        for imgs, _ in tqdm(loader, desc="통계 추출 중"):
            imgs = imgs.to(device)

            # 직접 forward pass 추적
            x = model.vit.patch_embed(imgs)
            cls = model.vit.cls_token.expand(imgs.shape[0], -1, -1)
            x = torch.cat([cls, x], dim=1)
            x = model.vit.pos_drop(x + model.vit.pos_embed)

            for l, block in enumerate(model.vit.blocks):
                # AdaMerge 적용 (실제 실행과 동일)
                x = model.ada_merge(x, layer_idx=l)

                # 이 시점의 patch 토큰으로 유사도 계산
                patch = x[:, 1:, :]  # CLS 제외
                T = patch.shape[1]

                if T >= 2:
                    # 이미지별로 수집
                    N = patch.shape[0]
                    for n in range(N):
                        sim = compute_feature_similarity_stats(
                            patch[n].unsqueeze(0)
                        )
                        stats_collector[l].append(sim)

                # Transformer block 실행
                x = block(x)

    # ── 통계 계산 및 저장 ─────────────────────
    stats = {}
    print("\n레이어별 유사도 통계 (AdaMerge 실행 환경 기준):")
    print(f"{'Layer':>6} {'avg_tokens':>12} {'μ':>8} {'σ':>8} {'min':>8} {'max':>8}")
    print("-" * 55)

    for l in range(num_layers):
        values = torch.tensor(stats_collector[l])
        mu    = values.mean().item()
        sigma = values.std().item()
        vmin  = values.min().item()
        vmax  = values.max().item()
        n_samples = len(values)
        stats[l] = {"mu": mu, "sigma": sigma}
        print(f"{l:>6} {n_samples:>12} {mu:>8.4f} {sigma:>8.4f} {vmin:>8.4f} {vmax:>8.4f}")

    stats_json = {str(k): v for k, v in stats.items()}
    with open(output_path, "w") as f:
        json.dump(stats_json, f, indent=2)

    print(f"\n통계 저장 완료: {output_path}")
    return stats


from stats_utils import load_stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",    type=str,   default="./data")
    parser.add_argument("--output",      type=str,   default="./stats.json")
    parser.add_argument("--prev_stats",  type=str,   default=None,
                        help="이전 stats.json 경로 (없으면 기본값 사용)")
    parser.add_argument("--subset",      type=float, default=0.01)
    parser.add_argument("--r_max",       type=int,   default=8)
    parser.add_argument("--alpha",       type=float, default=3.0)
    parser.add_argument("--batch_size",  type=int,   default=32)
    parser.add_argument("--device",      type=str,   default="cuda")
    args = parser.parse_args()

    precompute_stats(
        data_dir=args.data_dir,
        output_path=args.output,
        prev_stats_path=args.prev_stats,
        subset_ratio=args.subset,
        r_max=args.r_max,
        alpha=args.alpha,
        batch_size=args.batch_size,
        device=args.device,
    )