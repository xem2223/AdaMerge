"""
evaluate_tome_trainingfree_small.py
------------------------------------
ToMe Training-free 평가 스크립트 (ViT-Small 전용).
evaluate_tome_trainingfree.py의 ViT-Small 버전.

  - embed_dim: 384 (Base: 768)
  - num_heads:   6 (Base: 12)
  - num_layers: 12 (동일)

사용법:
  python evaluate_tome_trainingfree_small.py \
    --data_dir ./data \
    --r 4
"""

import argparse
import time
import json
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.cuda.amp import autocast
from tqdm import tqdm


# ─────────────────────────────────────────
# ToMe 핵심 로직 (Base 버전과 동일)
# ─────────────────────────────────────────

def bipartite_soft_matching(metric: torch.Tensor, r: int) -> torch.Tensor:
    N, T, D = metric.shape
    T_half = T // 2

    x_A = metric[:, :T_half]
    x_B = metric[:, T_half:]

    if r <= 0 or T_half <= r:
        return metric

    a_norm = F.normalize(x_A, dim=-1)
    b_norm = F.normalize(x_B, dim=-1)
    scores = torch.bmm(a_norm, b_norm.transpose(1, 2))  # [N, T/2, T/2]

    best_match = scores.argmax(dim=-1)
    top_scores = scores.amax(dim=-1)
    _, top_idx = top_scores.topk(r, dim=-1)

    merge_mask = torch.zeros(N, T_half, dtype=torch.bool, device=metric.device)
    merge_mask.scatter_(1, top_idx, True)

    b_idx = best_match.gather(1, top_idx)

    x_A_merge = x_A.gather(
        1, top_idx.unsqueeze(-1).expand(-1, -1, D)
    )
    x_B_target = x_B.gather(
        1, b_idx.unsqueeze(-1).expand(-1, -1, D)
    )

    x_merged = (x_A_merge + x_B_target) / 2

    x_B_out = x_B.clone()
    x_B_out.scatter_(
        1, b_idx.unsqueeze(-1).expand(-1, -1, D), x_merged
    )

    keep_mask = ~merge_mask
    x_A_keep = x_A[keep_mask].view(N, T_half - r, D)

    return torch.cat([x_A_keep, x_B_out], dim=1)


class ToMeBlock(nn.Module):
    def __init__(self, block: nn.Module, r: int):
        super().__init__()
        self.block = block
        self.r = r

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cls_token    = x[:, :1, :]
        patch_tokens = x[:, 1:, :]

        if self.r > 0:
            patch_tokens = bipartite_soft_matching(patch_tokens, self.r)

        x = torch.cat([cls_token, patch_tokens], dim=1)
        x = self.block(x)
        return x


def build_tome_model(r: int, pretrained: bool = True):
    model = timm.create_model("vit_small_patch16_224", pretrained=False)

    if pretrained:
        from safetensors.torch import load_file
        cache_path = os.path.expanduser(
            "~/.cache/huggingface/hub/"
            "models--timm--vit_small_patch16_224.augreg_in21k_ft_in1k/"
            "snapshots/7e2c55630205e1266030f18370f4c6ed1a514b52/model.safetensors"
        )
        state_dict = load_file(cache_path)
        model.load_state_dict(state_dict, strict=False)
        print(f"  가중치 로드 완료: {cache_path}")

    new_blocks = nn.Sequential(*[
        ToMeBlock(block, r) for block in model.blocks
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
        transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
        ),
    ])

    print("ImageNet-1k validation set 로드 중...")
    hf_data = load_dataset("ILSVRC/imagenet-1k", cache_dir=data_dir)
    val_dataset = HFImageNetDataset(hf_data["validation"], transform=val_transform)

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    print(f"Val: {len(val_dataset):,}장\n")
    return val_loader


# ─────────────────────────────────────────
# 정확도 / Throughput / FLOPs
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
    end = time.perf_counter()

    return (batch_size * n_runs) / (end - start)


@torch.no_grad()
def measure_token_and_flops(model, device, r, n_samples=200):
    model.eval()
    all_counts = []

    token_counts_per_sample = []

    def make_hook(l):
        def hook(module, input, output):
            token_counts_per_sample.append(output.shape[1])
        return hook

    hooks = []
    for block in model.blocks:
        hooks.append(block.register_forward_hook(make_hook(len(hooks))))

    for _ in range(n_samples):
        token_counts_per_sample.clear()
        dummy = torch.randn(1, 3, 224, 224).to(device)
        _ = model(dummy)
        all_counts.append([197] + token_counts_per_sample[:])

    for h in hooks:
        h.remove()

    n_layers = 12
    avg_counts = [
        sum(all_counts[i][l] for i in range(n_samples)) / n_samples
        for l in range(n_layers + 1)
    ]

    # FLOPs 추정 (ViT-Small: D=384)
    D = 384
    flops = 0.0
    for l in range(n_layers):
        T = avg_counts[l]
        flops += 2 * T * T * D + 4 * T * D * D  # Attention
        flops += 2 * T * D * (4 * D)              # MLP
    flops_g = flops / 1e9

    return avg_counts, flops_g


# ─────────────────────────────────────────
# 메인
# ─────────────────────────────────────────

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"ToMe r={args.r} / ViT-Small\n")

    val_loader = build_val_loader(args.data_dir, args.batch_size, args.num_workers)

    # ── 1. Baseline ViT-S ─────────────────
    print("=" * 55)
    print("[1/2] Baseline ViT-S/16 (pretrained, no merging)")
    print("=" * 55)

    from safetensors.torch import load_file
    baseline = timm.create_model("vit_small_patch16_224", pretrained=False)
    cache_path = os.path.expanduser(
        "~/.cache/huggingface/hub/"
        "models--timm--vit_small_patch16_224.augreg_in21k_ft_in1k/"
        "snapshots/7e2c55630205e1266030f18370f4c6ed1a514b52/model.safetensors"
    )
    state_dict = load_file(cache_path)
    baseline.load_state_dict(state_dict, strict=False)
    baseline = baseline.to(device)

    print("정확도 측정 중...")
    base_acc = evaluate(baseline, val_loader, device)
    base_thr = measure_throughput(baseline, device, args.batch_size)

    T, D = 197, 384  # ViT-Small: D=384
    base_flops = sum(
        2*T*T*D + 4*T*D*D + 2*T*D*(4*D)
        for _ in range(12)
    ) / 1e9

    print(f"  Top-1 Acc:  {base_acc:.2f}%")
    print(f"  FLOPs:      {base_flops:.2f} GFLOPs")
    print(f"  Throughput: {base_thr:.1f} img/s")

    del baseline
    torch.cuda.empty_cache()

    # ── 2. ToMe ViT-S (training-free) ─────
    print(f"\n{'='*55}")
    print(f"[2/2] ToMe ViT-S r={args.r} (pretrained + training-free)")
    print(f"{'='*55}")

    model = build_tome_model(r=args.r, pretrained=True).to(device)

    print("정확도 측정 중...")
    tome_acc = evaluate(model, val_loader, device)
    tome_thr = measure_throughput(model, device, args.batch_size)

    print(f"토큰 수 / FLOPs 측정 중 ({args.n_samples}샘플)...")
    avg_counts, tome_flops = measure_token_and_flops(
        model, device, args.r, args.n_samples
    )
    total_merged = avg_counts[0] - avg_counts[-1]

    print(f"  Top-1 Acc:       {tome_acc:.2f}%")
    print(f"  FLOPs:           {tome_flops:.2f} GFLOPs")
    print(f"  FLOPs 감소:      {(1-tome_flops/base_flops)*100:.1f}%")
    print(f"  Throughput:      {tome_thr:.1f} img/s")
    print(f"  Throughput 향상: {tome_thr/base_thr:.2f}×")
    print(f"  평균 병합 토큰:  {total_merged:.1f}개")

    # ── 3. 레이어별 토큰 수 ──────────────
    print(f"\n레이어별 평균 토큰 수:")
    for l, tc in enumerate(avg_counts):
        label = "입력" if l == 0 else f"Layer {l-1:2d}"
        bar = "█" * int(tc / avg_counts[0] * 25)
        print(f"  {label:>8}: {tc:5.1f}개  {bar}")

    # ── 4. 최종 요약 ─────────────────────
    print(f"\n{'='*55}")
    print(f"최종 요약 (Training-free, ToMe ViT-S r={args.r})")
    print(f"{'='*55}")
    print(f"{'Method':<25} {'Acc':>7} {'FLOPs':>9} {'Throughput':>12} {'속도':>8}")
    print(f"{'-'*65}")
    print(f"{'ViT-S/16 Baseline':<25} {base_acc:>6.2f}% {base_flops:>7.2f}G "
          f"{base_thr:>10.1f}/s {'1.00×':>8}")
    print(f"{f'ToMe ViT-S r={args.r} (TF)':<25} {tome_acc:>6.2f}% "
          f"{tome_flops:>7.2f}G {tome_thr:>10.1f}/s "
          f"{tome_thr/base_thr:>7.2f}×")

    # ── 5. 결과 저장 ─────────────────────
    result = {
        "setting": {
            "model": "vit_small_patch16_224",
            "r": args.r,
            "training_free": True,
            "method": "ToMe",
        },
        "baseline": {
            "acc": base_acc,
            "flops_g": base_flops,
            "throughput": base_thr,
        },
        "tome": {
            "acc": tome_acc,
            "flops_g": tome_flops,
            "flops_reduction_pct": (1 - tome_flops/base_flops)*100,
            "throughput": tome_thr,
            "throughput_speedup": tome_thr/base_thr,
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
    parser.add_argument("--r",           type=int,   default=4,
                        help="ToMe 레이어당 병합 수")
    parser.add_argument("--batch_size",  type=int,   default=64)
    parser.add_argument("--num_workers", type=int,   default=4)
    parser.add_argument("--n_samples",   type=int,   default=200)
    parser.add_argument("--output",      type=str,   default="./result_tome_trainingfree_small.json")
    args = parser.parse_args()
    main(args)