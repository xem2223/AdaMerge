"""
evaluate_trainingfree.py
------------------------
Training-free 평가 스크립트.

학습 없이 pretrained ViT에 AdaMerge를 바로 적용하고
ImageNet-1k val set에서 정확도, FLOPs, Throughput 측정.

사용법:
  python evaluate_trainingfree.py \
    --data_dir ./data \
    --stats    ./stats.json \
    --r_max    8 \
    --alpha    3.0
"""

import os
import argparse
import time
import json
import torch
import torch.nn as nn
import timm
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.cuda.amp import autocast
from tqdm import tqdm

from vit_adamerge import ViTWithAdaMerge


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
# 정확도 평가
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


# ─────────────────────────────────────────
# Throughput 측정
# ─────────────────────────────────────────

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


# ─────────────────────────────────────────
# 토큰 수 / FLOPs 측정
# ─────────────────────────────────────────

@torch.no_grad()
def measure_token_and_flops(model, device, n_samples=200):
    model.eval()
    all_counts = []

    for _ in range(n_samples):
        dummy = torch.randn(1, 3, 224, 224).to(device)
        x = model.vit.patch_embed(dummy)
        cls = model.vit.cls_token.expand(1, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = model.vit.pos_drop(x + model.vit.pos_embed)

        counts = [x.shape[1]]
        for l, block in enumerate(model.vit.blocks):
            x = model.ada_merge(x, layer_idx=l)
            x = block(x)
            counts.append(x.shape[1])
        all_counts.append(counts)

    n_layers = len(all_counts[0])
    avg_counts = [
        sum(all_counts[i][l] for i in range(n_samples)) / n_samples
        for l in range(n_layers)
    ]

    # FLOPs 추정
    D = 768
    flops = 0.0
    for l in range(n_layers - 1):
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
    print(f"설정: r_max={args.r_max}, alpha={args.alpha}\n")

    val_loader = build_val_loader(args.data_dir, args.batch_size, args.num_workers)

    # ── 1. Baseline ViT-B (학습 없음) ─────────
    print("=" * 55)
    print("[1/2] Baseline ViT-B/16 (pretrained, no merging)")
    print("=" * 55)

    from safetensors.torch import load_file
    import os
    baseline = timm.create_model("vit_base_patch16_224", pretrained=False)
    cache_path = os.path.expanduser(
        "~/.cache/huggingface/hub/"
        "models--timm--vit_base_patch16_224.augreg2_in21k_ft_in1k/"
        "snapshots/063c6c38a5d8510b2e57df480445e94b231dad2c/model.safetensors"
    )
    state_dict = load_file(cache_path)
    baseline.load_state_dict(state_dict, strict=False)
    baseline = baseline.to(device)

    print("정확도 측정 중...")
    base_acc = evaluate(baseline, val_loader, device)
    base_thr = measure_throughput(baseline, device, args.batch_size)

    T, D = 197, 768
    base_flops = sum(
        2*T*T*D + 4*T*D*D + 2*T*D*(4*D)
        for _ in range(12)
    ) / 1e9

    print(f"  Top-1 Acc:  {base_acc:.2f}%")
    print(f"  FLOPs:      {base_flops:.2f} GFLOPs")
    print(f"  Throughput: {base_thr:.1f} img/s")

    # ── 2. AdaMerge (학습 없음, training-free) ─
    print(f"\n{'='*55}")
    print(f"[2/2] AdaMerge (pretrained + training-free)")
    print(f"{'='*55}")

    model = ViTWithAdaMerge(
        r_max=args.r_max,
        alpha=args.alpha,
        stats_path=args.stats,
        pretrained=True,   # pretrained 가중치 사용
    ).to(device)
    # ← 학습 없이 바로 평가

    print("정확도 측정 중...")
    ada_acc = evaluate(model, val_loader, device)
    ada_thr = measure_throughput(model, device, args.batch_size)

    print("토큰 수 / FLOPs 측정 중...")
    avg_counts, ada_flops = measure_token_and_flops(model, device)

    total_merged = avg_counts[0] - avg_counts[-1]

    print(f"  Top-1 Acc:       {ada_acc:.2f}%")
    print(f"  FLOPs:           {ada_flops:.2f} GFLOPs")
    print(f"  FLOPs 감소:      {(1-ada_flops/base_flops)*100:.1f}%")
    print(f"  Throughput:      {ada_thr:.1f} img/s")
    print(f"  Throughput 향상: {ada_thr/base_thr:.2f}×")
    print(f"  평균 병합 토큰:  {total_merged:.1f}개")

    # ── 3. 레이어별 토큰 수 ───────────────────
    print(f"\n레이어별 평균 토큰 수:")
    for l, tc in enumerate(avg_counts):
        label = "입력" if l == 0 else f"Layer {l-1:2d}"
        bar = "█" * int(tc / avg_counts[0] * 25)
        print(f"  {label:>8}: {tc:5.1f}개  {bar}")

    # ── 4. 최종 요약 ──────────────────────────
    print(f"\n{'='*55}")
    print("최종 요약 (Training-free)")
    print(f"{'='*55}")
    print(f"{'Method':<25} {'Acc':>7} {'FLOPs':>9} {'Throughput':>12} {'속도':>8}")
    print(f"{'-'*65}")
    print(f"{'Baseline':<25} {base_acc:>6.2f}% {base_flops:>7.2f}G "
          f"{base_thr:>10.1f}/s {'1.00×':>8}")
    print(f"{'AdaMerge (training-free)':<25} {ada_acc:>6.2f}% {ada_flops:>7.2f}G "
          f"{ada_thr:>10.1f}/s {ada_thr/base_thr:>7.2f}×")

    # ── 5. 결과 저장 ──────────────────────────
    result = {
        "setting": {
            "r_max": args.r_max,
            "alpha": args.alpha,
            "training_free": True,
        },
        "baseline": {
            "acc": base_acc,
            "flops_g": base_flops,
            "throughput": base_thr,
        },
        "adamerge": {
            "acc": ada_acc,
            "flops_g": ada_flops,
            "flops_reduction_pct": (1 - ada_flops/base_flops)*100,
            "throughput": ada_thr,
            "throughput_speedup": ada_thr/base_thr,
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
    parser.add_argument("--stats",       type=str,   default="./stats.json")
    parser.add_argument("--r_max",       type=int,   default=8)
    parser.add_argument("--alpha",       type=float, default=3.0)
    parser.add_argument("--batch_size",  type=int,   default=64)
    parser.add_argument("--num_workers", type=int,   default=4)
    parser.add_argument("--output",      type=str,   default="./result_trainingfree.json")
    args = parser.parse_args()
    main(args)