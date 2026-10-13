"""
evaluate_dsm.py
---------------
DSM (Delayed Spatial Merging) Training-free 평가 스크립트.
논문: "Training-Free Acceleration of ViTs with Delayed Spatial Merging"
arXiv 2303.02331

DSM의 두 가지 핵심 요소:
  1. Delayed Merging:
     얕은 레이어에서는 attention이 아직 수렴 안 됨 (divergent attention)
     → 병합하면 오히려 성능 저하
     → delay_layer 이후부터만 병합

  2. Localized Merging (Hierarchical):
     ToMe는 전체 토큰에서 전역 탐색 → 먼 토큰끼리도 병합될 수 있음
     DSM은 로컬 윈도우 내에서만 탐색 → 공간적으로 가까운 토큰끼리 병합
     → 이웃 패치끼리 시맨틱 관계가 강하다는 원리 활용

사용법:
  python evaluate_dsm.py --data_dir ./data --r 8 --delay_layer 4
"""

import argparse
import time
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import math
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.cuda.amp import autocast
from tqdm import tqdm
import os
from safetensors.torch import load_file


# ─────────────────────────────────────────
# DSM 핵심 로직
# ─────────────────────────────────────────

def local_bipartite_merge(
    x: torch.Tensor,
    r: int,
    window_size: int = 4,
) -> torch.Tensor:
    """
    DSM의 Localized Bipartite Matching.

    전역 탐색(ToMe) 대신 로컬 윈도우 내에서만 유사한 토큰 쌍을 탐색.
    14×14 grid에서 window_size×window_size 윈도우로 나눠서 각 윈도우 내 병합.

    Args:
        x:           패치 토큰 [N, T, D]  (T=196, 14×14)
        r:           전체 병합할 토큰 수
        window_size: 로컬 윈도우 크기 (기본값: 4 → 4×4=16 토큰/윈도우)

    Returns:
        x_merged: [N, T-r, D]
    """
    N, T, D = x.shape

    if r <= 0:
        return x

    # 현재 토큰 수 T에서 H, W 계산
    # T가 완전제곱수가 아닐 수 있으므로 안전하게 처리
    H = int(math.sqrt(T))
    W = H
    # T가 H*W와 다르면 (병합 후 토큰이 줄어든 경우)
    # grid 분할 대신 1D로 처리
    if H * W != T:
        # 1D fallback: 전체를 하나의 윈도우로 처리
        T_half = T // 2
        r = min(r, T_half)
        if r <= 0:
            return x
        x_A = x[:, :T_half]
        x_B = x[:, T_half:]
        a_norm = F.normalize(x_A, dim=-1)
        b_norm = F.normalize(x_B, dim=-1)
        scores = torch.bmm(a_norm, b_norm.transpose(1, 2))
        best_match = scores.argmax(dim=-1)
        top_scores = scores.amax(dim=-1)
        _, top_idx = top_scores.topk(r, dim=-1)
        b_idx = best_match.gather(1, top_idx)
        x_A_merge = x_A.gather(1, top_idx.unsqueeze(-1).expand(-1,-1,D))
        x_B_target = x_B.gather(1, b_idx.unsqueeze(-1).expand(-1,-1,D))
        x_merged = (x_A_merge + x_B_target) / 2
        x_B_out = x_B.clone()
        x_B_out.scatter_(1, b_idx.unsqueeze(-1).expand(-1,-1,D), x_merged)
        merge_mask = torch.zeros(N, T_half, dtype=torch.bool, device=x.device)
        merge_mask.scatter_(1, top_idx, True)
        x_A_keep = x_A[~merge_mask].view(N, T_half - r, D)
        return torch.cat([x_A_keep, x_B_out], dim=1)

    # 윈도우 수: 올림으로 계산해서 모든 토큰이 포함되도록
    n_win_h = math.ceil(H / window_size)
    n_win_w = math.ceil(W / window_size)
    if n_win_h == 0 or n_win_w == 0:
        n_win_h = n_win_w = 1

    # 각 윈도우별 병합할 토큰 수 계산
    n_windows = n_win_h * n_win_w
    r_per_win = max(1, r // n_windows)

    # 결과 저장 버퍼
    x_2d = x.view(N, H, W, D)  # [N, H, W, D]
    keep_list = []

    for wh in range(n_win_h):
        for ww in range(n_win_w):
            h_start = wh * window_size
            h_end   = min(h_start + window_size, H)
            w_start = ww * window_size
            w_end   = min(w_start + window_size, W)

            # 윈도우 내 토큰 추출
            win_tokens = x_2d[:, h_start:h_end, w_start:w_end, :]
            wT = (h_end - h_start) * (w_end - w_start)
            win_flat = win_tokens.reshape(N, wT, D)

            if wT <= 2 or r_per_win <= 0:
                keep_list.append(win_flat)
                continue

            wT_half = wT // 2
            r_win = min(r_per_win, wT_half)

            x_A = win_flat[:, :wT_half]
            x_B = win_flat[:, wT_half:]

            a_norm = F.normalize(x_A, dim=-1)
            b_norm = F.normalize(x_B, dim=-1)
            scores = torch.bmm(a_norm, b_norm.transpose(1, 2))

            best_match  = scores.argmax(dim=-1)
            top_scores  = scores.amax(dim=-1)
            _, top_idx  = top_scores.topk(r_win, dim=-1)
            b_idx       = best_match.gather(1, top_idx)

            x_A_merge   = x_A.gather(1, top_idx.unsqueeze(-1).expand(-1,-1,D))
            x_B_target  = x_B.gather(1, b_idx.unsqueeze(-1).expand(-1,-1,D))
            x_merged    = (x_A_merge + x_B_target) / 2

            x_B_out = x_B.clone()
            x_B_out.scatter_(1, b_idx.unsqueeze(-1).expand(-1,-1,D), x_merged)

            merge_mask = torch.zeros(N, wT_half, dtype=torch.bool, device=x.device)
            merge_mask.scatter_(1, top_idx, True)
            x_A_keep = x_A[~merge_mask].view(N, wT_half - r_win, D)

            win_out = torch.cat([x_A_keep, x_B_out], dim=1)
            keep_list.append(win_out)

    return torch.cat(keep_list, dim=1)  # [N, T-r', D]


class DSMBlock(nn.Module):
    """
    DSM이 적용된 ViT Block.

    delay_layer 이전: 병합 없음 (divergent attention 구간)
    delay_layer 이후: localized bipartite matching으로 병합
    """
    def __init__(self, block: nn.Module, layer_idx: int,
                 delay_layer: int, r: int, window_size: int = 4):
        super().__init__()
        self.block       = block
        self.layer_idx   = layer_idx
        self.delay_layer = delay_layer
        self.r           = r
        self.window_size = window_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cls_token    = x[:, :1, :]
        patch_tokens = x[:, 1:, :]

        # delay_layer 이후부터만 localized merging
        if self.layer_idx >= self.delay_layer and self.r > 0:
            patch_tokens = local_bipartite_merge(
                patch_tokens, self.r, self.window_size
            )

        x = torch.cat([cls_token, patch_tokens], dim=1)
        x = self.block(x)
        return x


def build_dsm_model(r: int, delay_layer: int = 4,
                    window_size: int = 4, pretrained: bool = True):
    model = timm.create_model("vit_base_patch16_224", pretrained=False)

    if pretrained:
        cache_path = os.path.expanduser(
            "~/.cache/huggingface/hub/"
            "models--timm--vit_base_patch16_224.augreg2_in21k_ft_in1k/"
            "snapshots/063c6c38a5d8510b2e57df480445e94b231dad2c/model.safetensors"
        )
        model.load_state_dict(load_file(cache_path), strict=False)

    new_blocks = nn.Sequential(*[
        DSMBlock(block, l, delay_layer, r, window_size)
        for l, block in enumerate(model.blocks)
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

    loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    print(f"Val: {len(val_dataset):,}장\n")
    return loader


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

    avg_counts = [
        sum(all_counts[i][l] for i in range(n_samples)) / n_samples
        for l in range(13)
    ]
    D = 768
    flops = sum(
        2*T*T*D + 4*T*D*D + 2*T*D*(4*D)
        for T in avg_counts[:12]
    ) / 1e9
    return avg_counts, flops


# ─────────────────────────────────────────
# 메인
# ─────────────────────────────────────────

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"DSM r={args.r}, delay_layer={args.delay_layer}, "
          f"window_size={args.window_size}\n")

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

    # ── 2. DSM ───────────────────────────
    print(f"\n{'='*55}")
    print(f"[2/2] DSM (training-free)")
    print(f"{'='*55}")

    model = build_dsm_model(
        r=args.r, delay_layer=args.delay_layer,
        window_size=args.window_size, pretrained=True
    ).to(device)

    print("정확도 측정 중...")
    dsm_acc = evaluate(model, val_loader, device)
    dsm_thr = measure_throughput(model, device, args.batch_size)

    print(f"토큰 수 / FLOPs 측정 중 ({args.n_samples}샘플)...")
    avg_counts, dsm_flops = measure_token_and_flops(model, device, args.n_samples)
    total_merged = avg_counts[0] - avg_counts[-1]

    print(f"\n  Top-1 Acc:       {dsm_acc:.2f}%")
    print(f"  FLOPs:           {dsm_flops:.2f} GFLOPs")
    print(f"  FLOPs 감소:      {(1-dsm_flops/base_flops)*100:.1f}%")
    print(f"  Throughput:      {dsm_thr:.1f} img/s")
    print(f"  Throughput 향상: {dsm_thr/base_thr:.2f}×")
    print(f"  평균 병합 토큰:  {total_merged:.1f}개")

    print(f"\n레이어별 평균 토큰 수:")
    for l, tc in enumerate(avg_counts):
        label = "입력" if l == 0 else f"Layer {l-1:2d}"
        flag  = "" if l == 0 or (l-1) < args.delay_layer else "← 병합"
        bar   = "█" * int(tc / avg_counts[0] * 25)
        print(f"  {label:>8}: {tc:5.1f}개  {bar} {flag}")

    print(f"\n{'='*55}")
    print(f"최종 요약 (Training-free, DSM)")
    print(f"{'='*55}")
    print(f"{'Method':<35} {'Acc':>7} {'FLOPs':>9} {'속도':>8}")
    print(f"{'-'*62}")
    print(f"{'Baseline':<35} {base_acc:>6.2f}% {base_flops:>7.2f}G {'1.00×':>8}")
    print(f"{f'DSM r={args.r} d={args.delay_layer} w={args.window_size}':<35} "
          f"{dsm_acc:>6.2f}% {dsm_flops:>7.2f}G {dsm_thr/base_thr:>7.2f}×")

    result = {
        "setting": {
            "r": args.r, "delay_layer": args.delay_layer,
            "window_size": args.window_size,
            "training_free": True, "method": "DSM"
        },
        "baseline": {"acc": base_acc, "flops_g": base_flops, "throughput": base_thr},
        "dsm": {
            "acc": dsm_acc,
            "flops_g": dsm_flops,
            "flops_reduction_pct": (1 - dsm_flops/base_flops)*100,
            "throughput": dsm_thr,
            "throughput_speedup": dsm_thr/base_thr,
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
    parser.add_argument("--r",           type=int,   default=8)
    parser.add_argument("--delay_layer", type=int,   default=4)
    parser.add_argument("--window_size", type=int,   default=4,
                        help="로컬 윈도우 크기 (기본값: 4 → 4×4=16 토큰/윈도우)")
    parser.add_argument("--batch_size",  type=int,   default=64)
    parser.add_argument("--num_workers", type=int,   default=4)
    parser.add_argument("--n_samples",   type=int,   default=200)
    parser.add_argument("--output",      type=str,   default="./result_dsm.json")
    args = parser.parse_args()
    main(args)