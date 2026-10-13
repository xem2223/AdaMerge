"""
evaluate_pitome.py
------------------
PiToMe (NeurIPS 2024) Training-free 평가 스크립트.
논문: "Accelerating Transformers with Spectrum-Preserving Token Merging"

핵심 아이디어:
  1. 전체 토큰의 energy score 계산
     (다른 토큰들과 얼마나 유사한지의 합산)
  2. energy 기준 정렬 후 A/B 교대 배치
     (인접한 energy 토큰들이 서로 다른 그룹에 배치됨)
     → A: high-energy (병합 후보)
     → B: low-energy  (보존)
  3. A→B 코사인 유사도 계산 후 top-r 쌍 병합
  4. weighted average (energy 가중) 병합

ToMe와의 차이:
  - ToMe: 인덱스 홀짝으로 A/B 나눔 → 인접한 토큰끼리 잘못 매칭 가능
  - PiToMe: energy 기준 정렬 후 교대 배치
            → 유사한 energy 토큰끼리 경쟁적으로 매칭
            → 중요 토큰(low-energy) 보존
"""

import argparse
import time
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.cuda.amp import autocast
from tqdm import tqdm
import os
from safetensors.torch import load_file


# ─────────────────────────────────────────
# PiToMe 핵심 로직
# ─────────────────────────────────────────

def pitome_merge(x: torch.Tensor, r: int) -> torch.Tensor:
    """
    PiToMe 병합 (논문 방식 구현).

    Step 1. Energy score 계산
      각 토큰이 다른 토큰들과 얼마나 유사한지 측정.
      high-energy = 주변에 유사한 토큰 많음 = 중복 = 병합 대상
      low-energy  = 고립된 고유 토큰 = 보존 대상

    Step 2. Energy 기준 정렬 후 A/B 교대 배치
      정렬된 순서에서 0,2,4... → A (high-energy)
                                1,3,5... → B (low-energy)
      → 인접한 energy 값의 토큰들이 서로 다른 그룹에 배치
      → A 토큰이 유사한 energy의 B 토큰을 찾을 확률이 높아짐

    Step 3. A→B 코사인 유사도로 top-r 쌍 병합
      energy 가중 평균으로 병합

    Args:
        x: [N, T, D]
        r: 병합할 토큰 수

    Returns:
        x_merged: [N, T-r, D]
    """
    N, T, D = x.shape

    if r <= 0 or T <= r * 2:
        return x

    # ── Step 1. Energy score 계산 ─────────
    x_norm = F.normalize(x, dim=-1)  # [N, T, D]

    # 전체 유사도 행렬 → energy = 행 합산 (자기자신 제외)
    sim_all = torch.bmm(x_norm, x_norm.transpose(1, 2))  # [N, T, T]
    # 자기자신(대각선) 0으로
    eye = torch.eye(T, device=x.device, dtype=torch.bool).unsqueeze(0)
    sim_all = sim_all.masked_fill(eye, 0.0)
    energy = sim_all.sum(dim=-1)  # [N, T]

    # ── Step 2. Energy 기준 정렬 후 A/B 교대 배치 ──
    # energy 내림차순 정렬
    sorted_idx = energy.argsort(dim=-1, descending=True)  # [N, T]

    # 교대 배치: 짝수 위치 → A(high-energy), 홀수 위치 → B(low-energy)
    A_pos = torch.arange(0, T, 2, device=x.device)   # 0,2,4,...
    B_pos = torch.arange(1, T, 2, device=x.device)   # 1,3,5,...

    A_idx = sorted_idx[:, A_pos]   # [N, T//2] high-energy 토큰 인덱스
    B_idx = sorted_idx[:, B_pos]   # [N, T//2] low-energy  토큰 인덱스

    T_A = A_idx.shape[1]  # ceil(T/2), T 홀수면 T//2+1
    T_B = B_idx.shape[1]  # T//2

    # A, B 토큰 추출
    x_A = x.gather(1, A_idx.unsqueeze(-1).expand(-1, -1, D))  # [N, T_A, D]
    x_B = x.gather(1, B_idx.unsqueeze(-1).expand(-1, -1, D))  # [N, T_B, D]
    e_A = energy.gather(1, A_idx)  # [N, T_A]
    e_B = energy.gather(1, B_idx)  # [N, T_B]

    # ── Step 3. A→B 코사인 유사도로 top-r 쌍 선택 ──
    a_norm = F.normalize(x_A, dim=-1)
    b_norm = F.normalize(x_B, dim=-1)
    scores = torch.bmm(a_norm, b_norm.transpose(1, 2))  # [N, T_A, T_B]

    # 각 A 토큰의 가장 유사한 B 인덱스 (값 범위: [0, T_B))
    best_match = scores.argmax(dim=-1)   # [N, T_A]

    # 유사도 상위 r개 A 토큰 선택
    r = min(r, T_A)
    top_scores = scores.amax(dim=-1)     # [N, T_A]
    _, top_idx = top_scores.topk(r, dim=-1)  # [N, r], 값 범위: [0, T_A)

    b_idx_merge = best_match.gather(1, top_idx)  # [N, r], 값 범위: [0, T_B)

    # ── Step 4. Energy 가중 평균 병합 ───────────
    x_A_merge = x_A.gather(
        1, top_idx.unsqueeze(-1).expand(-1, -1, D)
    )  # [N, r, D]
    x_B_target = x_B.gather(
        1, b_idx_merge.unsqueeze(-1).expand(-1, -1, D)
    )  # [N, r, D]

    e_A_merge = e_A.gather(1, top_idx).unsqueeze(-1)          # [N, r, 1]
    e_B_target = e_B.gather(1, b_idx_merge).unsqueeze(-1)     # [N, r, 1]

    denom = e_A_merge + e_B_target + 1e-6
    x_merged_vals = (
        e_A_merge * x_A_merge + e_B_target * x_B_target
    ) / denom  # [N, r, D]

    # B에 병합 결과 반영
    x_B_out = x_B.clone()
    x_B_out.scatter_(
        1,
        b_idx_merge.unsqueeze(-1).expand(-1, -1, D),
        x_merged_vals
    )

    # 병합 안 된 A 토큰 보존 (merge_mask 크기를 T_A로)
    merge_mask = torch.zeros(N, T_A, dtype=torch.bool, device=x.device)
    merge_mask.scatter_(1, top_idx, True)  # top_idx 값 범위 [0, T_A) ✓
    keep_mask = ~merge_mask

    x_A_keep = x_A[keep_mask].view(N, T_A - r, D)

    return torch.cat([x_A_keep, x_B_out], dim=1)  # [N, T-r, D]


class PiToMeBlock(nn.Module):
    """PiToMe가 적용된 ViT Block."""

    def __init__(self, block: nn.Module, r: int):
        super().__init__()
        self.block = block
        self.r = r

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cls_token    = x[:, :1, :]
        patch_tokens = x[:, 1:, :]

        if self.r > 0:
            patch_tokens = pitome_merge(patch_tokens, self.r)

        x = torch.cat([cls_token, patch_tokens], dim=1)
        x = self.block(x)
        return x


def build_pitome_model(r: int, pretrained: bool = True):
    """timm ViT-B에 PiToMe 적용."""
    model = timm.create_model("vit_base_patch16_224", pretrained=False)

    if pretrained:
        cache_path = os.path.expanduser(
            "~/.cache/huggingface/hub/"
            "models--timm--vit_base_patch16_224.augreg2_in21k_ft_in1k/"
            "snapshots/063c6c38a5d8510b2e57df480445e94b231dad2c/model.safetensors"
        )
        state_dict = load_file(cache_path)
        model.load_state_dict(state_dict, strict=False)

    new_blocks = nn.Sequential(*[
        PiToMeBlock(block, r) for block in model.blocks
    ])
    model.blocks = new_blocks

    return model


# ─────────────────────────────────────────
# 데이터 로더
# ─────────────────────────────────────────

class HFImageNetDataset(torch.utils.data.Dataset):
    def __init__(self, hf_dataset, transform=None):
        self.dataset = hf_dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item["image"].convert("RGB")
        label = item["label"]
        if self.transform:
            image = self.transform(image)
        return image, label


def build_val_loader(data_dir, batch_size, num_workers):
    from datasets import load_dataset

    val_transform = transforms.Compose([
        transforms.Resize(249, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    print("ImageNet-1k validation set 로드 중...")
    hf_data = load_dataset("ILSVRC/imagenet-1k", cache_dir=data_dir)
    val_dataset = HFImageNetDataset(hf_data["validation"], transform=val_transform)

    return DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )


# ─────────────────────────────────────────
# 평가 / Throughput / FLOPs
# ─────────────────────────────────────────

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total   = 0
    for imgs, labels in tqdm(loader, desc="Evaluating"):
        imgs   = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with autocast():
            logits = model(imgs)
        pred = logits.argmax(dim=-1)
        correct += (pred == labels).sum().item()
        total   += labels.size(0)
    return correct / total * 100


@torch.no_grad()
def measure_throughput(model, device, batch_size=64, n_runs=50, warmup=10):
    model.eval()
    dummy = torch.randn(batch_size, 3, 224, 224).to(device)
    for _ in range(warmup):
        _ = model(dummy)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(n_runs):
        _ = model(dummy)
    torch.cuda.synchronize()
    return (batch_size * n_runs) / (time.perf_counter() - start)


@torch.no_grad()
def measure_token_and_flops(model, device, n_samples=200):
    model.eval()
    all_counts = []
    counts_buf = []

    def make_hook():
        def hook(module, input, output):
            counts_buf.append(output.shape[1])
        return hook

    hooks = [block.register_forward_hook(make_hook()) for block in model.blocks]

    for _ in range(n_samples):
        counts_buf.clear()
        _ = model(torch.randn(1, 3, 224, 224).to(device))
        all_counts.append([197] + counts_buf[:])

    for h in hooks:
        h.remove()

    n_layers = 12
    avg_counts = [
        sum(all_counts[i][l] for i in range(n_samples)) / n_samples
        for l in range(n_layers + 1)
    ]
    D = 768
    flops = sum(
        2*T*T*D + 4*T*D*D + 2*T*D*(4*D)
        for T in avg_counts[:n_layers]
    ) / 1e9

    return avg_counts, flops


# ─────────────────────────────────────────
# 메인
# ─────────────────────────────────────────

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"PiToMe r={args.r}\n")

    val_loader = build_val_loader(args.data_dir, args.batch_size, args.num_workers)

    # ── 1. Baseline ──────────────────────
    print("=" * 55)
    print("[1/2] Baseline ViT-B/16")
    print("=" * 55)

    baseline = timm.create_model("vit_base_patch16_224", pretrained=False)
    cache_path = os.path.expanduser(
        "~/.cache/huggingface/hub/"
        "models--timm--vit_base_patch16_224.augreg2_in21k_ft_in1k/"
        "snapshots/063c6c38a5d8510b2e57df480445e94b231dad2c/model.safetensors"
    )
    baseline.load_state_dict(load_file(cache_path), strict=False)
    baseline = baseline.to(device)

    print("정확도 측정 중...")
    base_acc = evaluate(baseline, val_loader, device)
    base_thr = measure_throughput(baseline, device, args.batch_size)
    T, D = 197, 768
    base_flops = sum(2*T*T*D + 4*T*D*D + 2*T*D*(4*D) for _ in range(12)) / 1e9

    print(f"  Top-1 Acc:  {base_acc:.2f}%")
    print(f"  FLOPs:      {base_flops:.2f} GFLOPs")
    print(f"  Throughput: {base_thr:.1f} img/s")

    del baseline
    torch.cuda.empty_cache()

    # ── 2. PiToMe ────────────────────────
    print(f"\n{'='*55}")
    print(f"[2/2] PiToMe r={args.r} (training-free)")
    print(f"{'='*55}")

    model = build_pitome_model(r=args.r, pretrained=True).to(device)

    print("정확도 측정 중...")
    pitome_acc = evaluate(model, val_loader, device)
    pitome_thr = measure_throughput(model, device, args.batch_size)

    print(f"토큰 수 / FLOPs 측정 중 ({args.n_samples}샘플)...")
    avg_counts, pitome_flops = measure_token_and_flops(model, device, args.n_samples)
    total_merged = avg_counts[0] - avg_counts[-1]

    print(f"\n  Top-1 Acc:       {pitome_acc:.2f}%")
    print(f"  FLOPs:           {pitome_flops:.2f} GFLOPs")
    print(f"  FLOPs 감소:      {(1-pitome_flops/base_flops)*100:.1f}%")
    print(f"  Throughput:      {pitome_thr:.1f} img/s")
    print(f"  Throughput 향상: {pitome_thr/base_thr:.2f}×")
    print(f"  평균 병합 토큰:  {total_merged:.1f}개")

    print(f"\n{'='*55}")
    print(f"최종 요약 (Training-free, PiToMe r={args.r})")
    print(f"{'='*55}")
    print(f"{'Method':<28} {'Acc':>7} {'FLOPs':>9} {'속도':>8}")
    print(f"{'-'*55}")
    print(f"{'Baseline':<28} {base_acc:>6.2f}% {base_flops:>7.2f}G {'1.00×':>8}")
    print(f"{f'PiToMe r={args.r}':<28} {pitome_acc:>6.2f}% "
          f"{pitome_flops:>7.2f}G {pitome_thr/base_thr:>7.2f}×")

    result = {
        "setting": {"r": args.r, "training_free": True, "method": "PiToMe"},
        "baseline": {"acc": base_acc, "flops_g": base_flops, "throughput": base_thr},
        "pitome": {
            "acc": pitome_acc,
            "flops_g": pitome_flops,
            "flops_reduction_pct": (1 - pitome_flops/base_flops)*100,
            "throughput": pitome_thr,
            "throughput_speedup": pitome_thr/base_thr,
            "avg_token_counts": avg_counts,
            "total_merged": total_merged,
        }
    }
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n결과 저장: {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",    type=str,   default="./data")
    parser.add_argument("--r",           type=int,   default=4)
    parser.add_argument("--batch_size",  type=int,   default=64)
    parser.add_argument("--num_workers", type=int,   default=4)
    parser.add_argument("--n_samples",   type=int,   default=200)
    parser.add_argument("--output",      type=str,   default="./result_pitome.json")
    args = parser.parse_args()
    main(args)