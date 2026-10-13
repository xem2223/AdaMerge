"""
convergence_experiment.py
--------------------------
수렴 곡선 + 초기화 robustness 실험.

세 가지 초기화에서 iterative refinement를 반복하며
각 iteration의 L2 norm 변화를 추적하고 figure 생성.

사용법:
  python convergence_experiment.py \
    --data_dir   ./data/imagenet1k \
    --r_max      16 \
    --max_iter   6 \
    --output_dir ./convergence_results
"""

import os
import json
import math
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pyarrow as pa
import glob
import io
from PIL import Image

from salience import compute_token_salience
from adamerge import compute_salience_weighted_similarity
from vit_adamerge import ViTWithAdaMerge


# ─────────────────────────────────────────
# 데이터 로드 (Arrow 포맷)
# ─────────────────────────────────────────

def load_arrow_subset(data_dir: str, subset_ratio: float = 0.01, seed: int = 42):
    """Arrow 파일에서 train subset 로드."""
    arrow_dir = os.path.join(
        data_dir,
        "ILSVRC___imagenet-1k", "default", "0.0.0",
        "49e2ee26f3810fb5a7536bbf732a7b07389a47b5",
    )
    train_files = sorted(glob.glob(
        os.path.join(arrow_dir, "imagenet-1k-train-*.arrow")
    ))
    if not train_files:
        raise FileNotFoundError(f"Train arrow 파일 없음: {arrow_dir}")

    print(f"  Arrow 파일 {len(train_files)}개 로드 중...")
    tables = []
    for f in train_files[:50]:  # 약 250K장
        with open(f, "rb") as src:
            reader = pa.ipc.open_stream(src)
            tables.append(reader.read_all())

    table = pa.concat_tables(tables)
    total = len(table)
    n_subset = int(total * subset_ratio)

    rng = np.random.RandomState(seed)
    indices = rng.choice(total, n_subset, replace=False)
    print(f"  총 {total}장 중 {n_subset}장 샘플링 (seed={seed})")
    return table, indices


class ArrowDataset(torch.utils.data.Dataset):
    def __init__(self, table, indices, transform=None):
        self.table   = table
        self.indices = indices
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        row = self.table.slice(int(self.indices[idx]), 1)
        img_data = row.column("image")[0].as_py()
        if isinstance(img_data, dict):
            img = Image.open(io.BytesIO(img_data["bytes"])).convert("RGB")
        else:
            img = Image.open(io.BytesIO(img_data)).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, 0


# ─────────────────────────────────────────
# 한 iteration: stats 추출
# ─────────────────────────────────────────

def extract_stats_one_iter(
    loader, model, num_layers: int, device
) -> dict:
    """모델 현재 stats로 forward하며 새 stats 수집."""
    collector = {l: [] for l in range(num_layers)}

    model.eval()
    with torch.no_grad():
        for imgs, _ in loader:
            imgs = imgs.to(device)

            x = model.vit.patch_embed(imgs)
            cls = model.vit.cls_token.expand(imgs.shape[0], -1, -1)
            x = torch.cat([cls, x], dim=1)
            x = model.vit.pos_drop(x + model.vit.pos_embed)

            for l, block in enumerate(model.vit.blocks):
                x = model.ada_merge(x, layer_idx=l)
                patch = x[:, 1:, :]
                T = patch.shape[1]
                if T >= 2:
                    T_half = T // 2
                    x_A = patch[:, :T_half]
                    x_B = patch[:, T_half:]
                    S   = compute_token_salience(patch)
                    sim = compute_salience_weighted_similarity(x_A, x_B, S)
                    max_sim = sim.amax(dim=-1).mean()
                    collector[l].append(max_sim.item())
                x = block(x)

    new_stats = {}
    for l in range(num_layers):
        vals = torch.tensor(collector[l])
        new_stats[l] = {
            "mu":    vals.mean().item(),
            "sigma": max(vals.std().item(), 1e-4),
        }
    return new_stats


def stats_l2_diff(stats_a: dict, stats_b: dict) -> float:
    """두 stats 간 L2 norm."""
    diffs = []
    for l in stats_a:
        diffs.append((stats_a[l]["mu"]    - stats_b[l]["mu"])    ** 2)
        diffs.append((stats_a[l]["sigma"] - stats_b[l]["sigma"]) ** 2)
    return math.sqrt(sum(diffs))


# ─────────────────────────────────────────
# Iterative refinement
# ─────────────────────────────────────────

def run_refinement(
    loader, init_stats: dict, r_max: int, alpha: float,
    max_iter: int, eps: float, num_layers: int, device, label: str
) -> tuple:
    """
    Returns:
        final_stats: dict
        l2_history:  list of float (per iteration)
    """
    print(f"\n[{label}] 수렴 실험 시작")
    model = ViTWithAdaMerge(
        r_max=r_max, alpha=alpha, stats_path=None, pretrained=True
    ).to(device)
    model.ada_merge.stats = {
        l: {"mu": init_stats[l]["mu"], "sigma": init_stats[l]["sigma"]}
        for l in range(num_layers)
    }

    current_stats = {l: dict(init_stats[l]) for l in range(num_layers)}
    l2_history = []

    for k in range(1, max_iter + 1):
        new_stats = extract_stats_one_iter(loader, model, num_layers, device)
        diff = stats_l2_diff(current_stats, new_stats)
        l2_history.append(diff)

        print(f"  iter {k}: ‖Δstats‖₂ = {diff:.6f}")

        # 모델 stats 업데이트
        model.ada_merge.stats = {
            l: {"mu": new_stats[l]["mu"], "sigma": new_stats[l]["sigma"]}
            for l in range(num_layers)
        }
        current_stats = new_stats

        if diff < eps:
            print(f"  → 수렴 (eps={eps})")
            break

    return current_stats, l2_history


# ─────────────────────────────────────────
# Figure 생성
# ─────────────────────────────────────────

def make_convergence_figure(results: dict, output_path: str):
    """
    results: {label: l2_history list}
    """
    fig, ax = plt.subplots(figsize=(5.5, 3.5))

    styles = {
        "Init-A (μ=0.5, σ=0.10)": ("o-", "#185FA5"),
        "Init-B (μ=0.3, σ=0.05)": ("s--", "#0F6E56"),
        "Init-C (μ=0.7, σ=0.15)": ("^:", "#854F0B"),
    }

    for label, history in results.items():
        style, color = styles.get(label, ("o-", "gray"))
        iters = list(range(1, len(history) + 1))
        ax.semilogy(iters, history, style, color=color,
                    label=label, linewidth=1.5, markersize=5)

    ax.set_xlabel("Iteration $k$", fontsize=11)
    ax.set_ylabel(r"$\|\Delta\,\mathrm{stats}^{(k)}\|_2$", fontsize=11)
    ax.set_title("Iterative Statistics Refinement Convergence", fontsize=11)
    ax.legend(fontsize=9, framealpha=0.9)
    ax.grid(True, which="both", alpha=0.3, linewidth=0.5)
    ax.set_xticks(range(1, max(len(h) for h in results.values()) + 1))

    plt.tight_layout()
    plt.savefig(output_path, format="pdf", dpi=300, bbox_inches="tight")
    plt.savefig(output_path.replace(".pdf", ".png"),
                format="png", dpi=200, bbox_inches="tight")
    plt.close()
    print(f"\n[Figure] 저장: {output_path}")


# ─────────────────────────────────────────
# Main
# ─────────────────────────────────────────

def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[장치] {device}")

    # 데이터 로드
    transform = transforms.Compose([
        transforms.Resize(249, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    table, indices = load_arrow_subset(args.data_dir, args.subset)
    dataset = ArrowDataset(table, indices, transform=transform)
    loader  = DataLoader(dataset, batch_size=args.batch_size,
                         shuffle=False, num_workers=4, pin_memory=True)

    num_layers = 12

    # 세 가지 초기화
    inits = {
        "Init-A (μ=0.5, σ=0.10)": {
            l: {"mu": 0.5, "sigma": 0.10} for l in range(num_layers)
        },
        "Init-B (μ=0.3, σ=0.05)": {
            l: {"mu": 0.3, "sigma": 0.05} for l in range(num_layers)
        },
        "Init-C (μ=0.7, σ=0.15)": {
            l: {"mu": 0.7, "sigma": 0.15} for l in range(num_layers)
        },
    }

    results     = {}  # label → l2_history
    final_stats = {}  # label → final stats dict

    for label, init in inits.items():
        stats, history = run_refinement(
            loader=loader,
            init_stats=init,
            r_max=args.r_max,
            alpha=args.alpha,
            max_iter=args.max_iter,
            eps=args.eps,
            num_layers=num_layers,
            device=device,
            label=label,
        )
        results[label]     = history
        final_stats[label] = stats

    # 수렴한 stats 비교 (레이어 0, 5, 11)
    print("\n=== Final Stats Comparison ===")
    print(f"{'Init':40s} | l=0 μ  | l=5 μ  | l=11 μ")
    print("-" * 75)
    for label, stats in final_stats.items():
        print(f"{label:40s} | {stats[0]['mu']:.4f} | "
              f"{stats[5]['mu']:.4f} | {stats[11]['mu']:.4f}")

    # L2 diff between Init-A and others (terminal stats)
    print("\n=== Terminal Stats L2 diff vs Init-A ===")
    ref = final_stats["Init-A (μ=0.5, σ=0.10)"]
    for label, stats in final_stats.items():
        diff = stats_l2_diff(ref, stats)
        print(f"  {label}: {diff:.6f}")

    # Figure 저장
    fig_path = os.path.join(args.output_dir, "convergence_curves.pdf")
    make_convergence_figure(results, fig_path)

    # 결과 JSON 저장
    out = {
        "l2_histories": {k: v for k, v in results.items()},
        "final_stats_sample": {
            label: {
                str(l): stats[l]
                for l in [0, 5, 11]
            }
            for label, stats in final_stats.items()
        }
    }
    with open(os.path.join(args.output_dir, "convergence_results.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"[결과] {args.output_dir}/convergence_results.json 저장 완료")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",    type=str,   default="./data/imagenet1k")
    parser.add_argument("--r_max",       type=int,   default=16)
    parser.add_argument("--alpha",       type=float, default=3.0)
    parser.add_argument("--subset",      type=float, default=0.01)
    parser.add_argument("--max_iter",    type=int,   default=6)
    parser.add_argument("--eps",         type=float, default=0.005)
    parser.add_argument("--batch_size",  type=int,   default=64)
    parser.add_argument("--output_dir",  type=str,   default="./convergence_results")
    args = parser.parse_args()
    main(args)