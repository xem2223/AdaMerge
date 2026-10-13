"""
ablation_experiments.py
------------------------
두 가지 ablation 실험:
  1. Subset-Size Robustness: stats 추출 subset 크기별 정확도
  2. Sigmoid Temperature:    adaptive-r temperature 민감도

사용법:
  # Subset-Size Robustness
  python ablation_experiments.py subset \
    --data_dir  ./data/imagenet1k \
    --r_max     12 \
    --output    ./ablation_results/subset_size.json

  # Sigmoid Temperature
  python ablation_experiments.py temperature \
    --data_dir   ./data/imagenet1k \
    --stats_path ./stats_rmax12_augreg2.json \
    --r_max      12 \
    --output     ./ablation_results/temperature.json
"""

import os
import sys
import json
import glob
import io
import math
import time
import random
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast
import pyarrow as pa
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

from salience import compute_token_salience
from adamerge import AdaMerge, compute_salience_weighted_similarity
from vit_adamerge import ViTWithAdaMerge
from stats_utils import load_stats


# ─────────────────────────────────────────
# 데이터 로드
# ─────────────────────────────────────────

TRANSFORM = transforms.Compose([
    transforms.Resize(249, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])


class ArrowDataset(torch.utils.data.Dataset):
    def __init__(self, table, indices=None, transform=None):
        self.table     = table
        self.indices   = indices if indices is not None else list(range(len(table)))
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        row      = self.table.slice(int(self.indices[i]), 1)
        img_data = row.column("image")[0].as_py()
        label    = row.column("label")[0].as_py()
        if isinstance(img_data, dict):
            img = Image.open(io.BytesIO(img_data["bytes"])).convert("RGB")
        else:
            img = Image.open(io.BytesIO(img_data)).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


def load_arrow_table(data_dir: str, split: str = "validation"):
    arrow_dir = os.path.join(
        data_dir,
        "ILSVRC___imagenet-1k", "default", "0.0.0",
        "49e2ee26f3810fb5a7536bbf732a7b07389a47b5",
    )
    files = sorted(glob.glob(
        os.path.join(arrow_dir, f"imagenet-1k-{split}-*.arrow")
    ))
    if not files:
        raise FileNotFoundError(f"{split} arrow 파일 없음: {arrow_dir}")

    tables = []
    for f in files:
        with open(f, "rb") as src:
            reader = pa.ipc.open_stream(src)
            tables.append(reader.read_all())
    return pa.concat_tables(tables)


def load_train_subset(data_dir: str, subset_ratio: float, seed: int = 42):
    arrow_dir = os.path.join(
        data_dir,
        "ILSVRC___imagenet-1k", "default", "0.0.0",
        "49e2ee26f3810fb5a7536bbf732a7b07389a47b5",
    )
    train_files = sorted(glob.glob(
        os.path.join(arrow_dir, "imagenet-1k-train-*.arrow")
    ))
    tables = []
    for f in train_files:
        with open(f, "rb") as src:
            reader = pa.ipc.open_stream(src)
            tables.append(reader.read_all())
    table = pa.concat_tables(tables)

    total    = len(table)
    n_subset = int(total * subset_ratio)
    rng      = np.random.RandomState(seed)
    indices  = rng.choice(total, n_subset, replace=False)
    return table, indices


# ─────────────────────────────────────────
# 평가 함수
# ─────────────────────────────────────────

@torch.no_grad()
def evaluate(model, loader, device) -> float:
    model.eval()
    correct, total = 0, 0
    for imgs, labels in tqdm(loader, desc="Evaluating", leave=False):
        imgs   = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with autocast():
            logits = model(imgs)
        pred     = logits.argmax(dim=-1)
        correct += (pred == labels).sum().item()
        total   += labels.size(0)
    return correct / total * 100


# ─────────────────────────────────────────
# Stats 추출 (단순 single-pass)
# ─────────────────────────────────────────

def extract_stats_simple(
    model, data_dir: str, subset_ratio: float,
    num_layers: int, device, seed: int = 42
) -> dict:
    """한 번의 forward pass로 stats 추출."""
    table, indices = load_train_subset(data_dir, subset_ratio, seed)
    dataset = ArrowDataset(table, indices, transform=TRANSFORM)
    loader  = DataLoader(dataset, batch_size=64, shuffle=False,
                         num_workers=4, pin_memory=True)

    collector = {l: [] for l in range(num_layers)}
    model.eval()

    with torch.no_grad():
        for imgs, _ in tqdm(loader,
                            desc=f"Stats 추출 ({subset_ratio*100:.1f}%)",
                            leave=False):
            imgs = imgs.to(device)

            x = model.vit.patch_embed(imgs)
            cls = model.vit.cls_token.expand(imgs.shape[0], -1, -1)
            x = torch.cat([cls, x], dim=1)
            x = model.vit.pos_drop(x + model.vit.pos_embed)

            for l, block in enumerate(model.vit.blocks):
                x = model.ada_merge(x, layer_idx=l)
                patch = x[:, 1:]
                T = patch.shape[1]
                if T >= 2:
                    T_half = T // 2
                    x_A = patch[:, :T_half]
                    x_B = patch[:, T_half:]
                    S   = compute_token_salience(patch)
                    sim = compute_salience_weighted_similarity(x_A, x_B, S)
                    collector[l].append(sim.amax(dim=-1).mean().item())
                x = block(x)

    stats = {}
    for l in range(num_layers):
        vals = torch.tensor(collector[l])
        stats[l] = {
            "mu":    vals.mean().item(),
            "sigma": max(vals.std().item(), 1e-4),
        }
    return stats


# ─────────────────────────────────────────
# Experiment 1: Subset-Size Robustness
# ─────────────────────────────────────────

def run_subset_size(args):
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Subset-Size Robustness] r_max={args.r_max}")

    # Val loader (고정)
    val_table  = load_arrow_table(args.data_dir, "validation")
    val_dataset = ArrowDataset(val_table, transform=TRANSFORM)
    val_loader  = DataLoader(val_dataset, batch_size=args.batch_size,
                             shuffle=False, num_workers=4, pin_memory=True)

    subset_ratios = [0.005, 0.01, 0.02, 0.05]  # 0.5%, 1%, 2%, 5%
    results = {}

    for ratio in subset_ratios:
        n_images = int(1281000 * ratio)  # 약 ImageNet train 크기 기준
        label    = f"{ratio*100:.1f}%"
        print(f"\n── Subset {label} (≈{n_images}장) ──")

        # 모델 로드 (기본 stats로 시작)
        model = ViTWithAdaMerge(
            r_max=args.r_max, alpha=args.alpha,
            stats_path=args.prev_stats,   # ← 수렴된 stats 사용
            pretrained=True,
        ).to(device)

        # Stats 추출
        stats = extract_stats_simple(
            model, args.data_dir, ratio, 12, device
        )

        # 모델에 stats 적용
        model.ada_merge.stats = {
            l: {"mu": stats[l]["mu"], "sigma": stats[l]["sigma"]}
            for l in range(12)
        }

        # 평가
        acc = evaluate(model, val_loader, device)
        print(f"  Top-1 Acc: {acc:.2f}%")
        results[label] = {
            "subset_ratio":  ratio,
            "n_images_approx": n_images,
            "top1_acc":      round(acc, 2),
        }

    print("\n=== Subset-Size Results ===")
    print(f"{'Subset':>8} | {'N images':>10} | {'Top-1 Acc':>10}")
    print("-" * 35)
    for label, r in results.items():
        print(f"{label:>8} | {r['n_images_approx']:>10} | {r['top1_acc']:>10.2f}%")

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[저장] {args.output}")


# ─────────────────────────────────────────
# Experiment 2: Sigmoid Temperature
# ─────────────────────────────────────────

class AdaMergeWithTemp(AdaMerge):
    """Temperature 파라미터를 추가한 AdaMerge."""

    def __init__(self, temperature: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.temperature = temperature

    def forward(self, x: torch.Tensor, layer_idx: int) -> torch.Tensor:
        from adamerge import compute_adaptive_r, bipartite_merge
        import math

        cls_token    = x[:, :1, :]
        patch_tokens = x[:, 1:, :]
        T = patch_tokens.shape[1]
        T_half = T // 2

        S   = compute_token_salience(patch_tokens)
        x_A = patch_tokens[:, :T_half]
        x_B = patch_tokens[:, T_half:]
        sim = compute_salience_weighted_similarity(x_A, x_B, S)

        mu_l    = self.stats[layer_idx]["mu"]
        sigma_l = self.stats[layer_idx]["sigma"]

        # Temperature 적용: z_l → z_l / T
        max_sim = sim.amax(dim=-1)
        S_bar   = max_sim.mean().item()
        if not math.isfinite(S_bar):
            r = self.r_max // 2
        else:
            sigma_safe = max(sigma_l, 1e-4)
            z          = (S_bar - mu_l) / sigma_safe / self.temperature
            if not math.isfinite(z):
                r = self.r_max // 2
            else:
                sig  = torch.sigmoid(torch.tensor(
                    self.alpha * z, dtype=torch.float32
                )).item()
                r = math.floor(self.r_max * sig)
                r = max(0, min(r, self.r_max))

        patch_merged, _ = bipartite_merge(patch_tokens, S, r, sim)
        return torch.cat([cls_token, patch_merged], dim=1)


def run_temperature(args):
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Sigmoid Temperature] r_max={args.r_max}, stats={args.stats_path}")

    stats = load_stats(args.stats_path)

    val_table   = load_arrow_table(args.data_dir, "validation")
    val_dataset = ArrowDataset(val_table, transform=TRANSFORM)
    val_loader  = DataLoader(val_dataset, batch_size=args.batch_size,
                             shuffle=False, num_workers=4, pin_memory=True)

    temperatures = [0.5, 1.0, 2.0, 4.0]
    results = {}

    for T in temperatures:
        label = f"T={T}"
        print(f"\n── Temperature {label} ──")

        # ViT 로드 + AdaMergeWithTemp 교체
        model = ViTWithAdaMerge(
            r_max=args.r_max, alpha=args.alpha,
            stats_path=args.stats_path, pretrained=True,
        ).to(device)

        # AdaMerge → AdaMergeWithTemp 교체
        model.ada_merge = AdaMergeWithTemp(
            temperature=T,
            num_layers=12,
            r_max=args.r_max,
            alpha=args.alpha,
            stats=stats,
        ).to(device)

        acc = evaluate(model, val_loader, device)
        print(f"  Top-1 Acc: {acc:.2f}%")
        results[label] = {
            "temperature": T,
            "top1_acc":    round(acc, 2),
        }

    print("\n=== Temperature Results ===")
    print(f"{'Temp':>8} | {'Top-1 Acc':>10}")
    print("-" * 22)
    for label, r in results.items():
        print(f"{label:>8} | {r['top1_acc']:>10.2f}%")

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[저장] {args.output}")


# ─────────────────────────────────────────
# Main
# ─────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="experiment")

    # Subset-Size
    p1 = sub.add_parser("subset")
    p1.add_argument("--data_dir",   type=str,   required=True)
    p1.add_argument("--r_max",      type=int,   default=12)
    p1.add_argument("--alpha",      type=float, default=3.0)
    p1.add_argument("--batch_size", type=int,   default=128)
    p1.add_argument("--output",     type=str,
                    default="./ablation_results/subset_size.json")
    p1.add_argument("--prev_stats", type=str, required=True,
                    help="수렴된 stats.json 경로 (시작점)")
    
    # Temperature
    p2 = sub.add_parser("temperature")
    p2.add_argument("--data_dir",    type=str,   required=True)
    p2.add_argument("--stats_path",  type=str,   required=True)
    p2.add_argument("--r_max",       type=int,   default=12)
    p2.add_argument("--alpha",       type=float, default=3.0)
    p2.add_argument("--batch_size",  type=int,   default=128)
    p2.add_argument("--output",      type=str,
                    default="./ablation_results/temperature.json")

    args = parser.parse_args()

    if args.experiment == "subset":
        run_subset_size(args)
    elif args.experiment == "temperature":
        run_temperature(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()