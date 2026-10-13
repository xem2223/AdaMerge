"""
ablation_component.py  (v3 — monkey-patch 제거, 이미지 단위 stats 수집)
------------------------------------------------------------------------
Component-wise ablation:

  SW  = Salience-Weighted similarity
  Adp = Adaptive r decision

  ┌──────────────────────────┬──────┬──────┬─────────────────────────────┐
  │ Config                   │  SW  │  Adp │ Stats used for r            │
  ├──────────────────────────┼──────┼──────┼─────────────────────────────┤
  │ ToMe (baseline)          │  ✗   │  ✗   │ N/A  (fixed r)              │
  │ Salience only            │  ✓   │  ✗   │ N/A  (fixed r)              │
  │ Adaptive r only          │  ✗   │  ✓   │ plain-cosine stats (새로 추출)│
  │ Full AdaMerge            │  ✓   │  ✓   │ salience-weighted stats     │
  └──────────────────────────┴──────┴──────┴─────────────────────────────┘

변경사항 (v3):
  - extract_plain_cosine_stats_clean(): monkey-patch 없이 ViT 직접 추적
  - 이미지 단위로 S_bar 수집 → sigma가 precompute_stats.py와 동일한 범위
  - _tome_merge_batch(): 배치 단위 ToMe 병합 헬퍼

사용법:
  python ablation_component.py \
    --data_dir      ./data/imagenet1k \
    --stats_path    ./stats_rmax17_augreg2.json \
    --r_max         17 \
    --tome_r        6 \
    --batch_size    128 \
    --output        ./ablation_results/component.json
"""

import os
import json
import glob
import io
import math
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
from adamerge import (
    AdaMerge,
    compute_salience_weighted_similarity,
    compute_adaptive_r,
    bipartite_merge,
)
from vit_adamerge import ViTWithAdaMerge
from stats_utils import load_stats


# ─────────────────────────────────────────
# 데이터셋
# ─────────────────────────────────────────

TRANSFORM = transforms.Compose([
    transforms.Resize(249, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])


class ArrowDataset(torch.utils.data.Dataset):
    def __init__(self, table, indices=None, transform=None):
        self.table   = table
        self.indices = list(range(len(table))) if indices is None else indices
        self.transform = transform

    def __len__(self): return len(self.indices)

    def __getitem__(self, i):
        row      = self.table.slice(int(self.indices[i]), 1)
        img_data = row.column("image")[0].as_py()
        label    = row.column("label")[0].as_py()
        img = Image.open(io.BytesIO(
            img_data["bytes"] if isinstance(img_data, dict) else img_data
        )).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


def load_arrow(data_dir, split):
    arrow_dir = os.path.join(
        data_dir,
        "ILSVRC___imagenet-1k", "default", "0.0.0",
        "49e2ee26f3810fb5a7536bbf732a7b07389a47b5",
    )
    files = sorted(glob.glob(
        os.path.join(arrow_dir, f"imagenet-1k-{split}-*.arrow")
    ))
    tables = []
    for f in files:
        with open(f, "rb") as src:
            reader = pa.ipc.open_stream(src)
            tables.append(reader.read_all())
    return pa.concat_tables(tables)


# ─────────────────────────────────────────
# ToMe 방식 병합 헬퍼 (배치 단위, salience 없음)
# ─────────────────────────────────────────

def _tome_merge_batch(patches: torch.Tensor,
                      sim: torch.Tensor,
                      r: int) -> torch.Tensor:
    """
    plain cosine sim 기반 ToMe 병합 (배치 단위).

    Args:
        patches: [B, T, D]
        sim:     [B, T/2, T/2]  — plain cosine similarity
        r:       병합할 토큰 수

    Returns:
        merged patches: [B, T-r, D]
    """
    B, T, D = patches.shape
    T_half  = T // 2
    r = min(r, T_half)
    if r == 0:
        return patches

    x_A = patches[:, :T_half]
    x_B = patches[:, T_half:]

    # Top-r 쌍 선택
    best_sim = sim.amax(dim=-1)                              # [B, T/2]
    _, top_i = best_sim.topk(r, dim=-1)                     # [B, r]
    j_idx    = sim.argmax(dim=-1).gather(1, top_i)          # [B, r]

    # 비가중 산술 평균
    xA_top = x_A.gather(1, top_i.unsqueeze(-1).expand(-1, -1, D))
    xB_tgt = x_B.gather(1, j_idx.unsqueeze(-1).expand(-1, -1, D))
    x_B    = x_B.clone()
    x_B.scatter_(
        1,
        j_idx.unsqueeze(-1).expand(-1, -1, D),
        (xA_top + xB_tgt) / 2.0,
    )

    # 병합되지 않은 A 토큰 유지
    merge_mask = torch.zeros(B, T_half, dtype=torch.bool, device=patches.device)
    merge_mask.scatter_(1, top_i, True)
    x_A_keep = x_A[~merge_mask].view(B, T_half - r, D)

    return torch.cat([x_A_keep, x_B], dim=1)


# ─────────────────────────────────────────
# Plain-cosine stats 추출 (monkey-patch 없음)
# ─────────────────────────────────────────

@torch.no_grad()
def extract_plain_cosine_stats_clean(
    vit,
    prev_stats: dict,
    data_dir: str,
    subset_ratio: float,
    r_max: int,
    alpha: float,
    num_layers: int = 12,
    device: str = "cuda",
    seed: int = 42,
) -> dict:
    """
    precompute_stats.py 방식으로 plain cosine stats 추출.

    monkey-patch 없이 ViT blocks를 직접 추적한다.
    이미지 단위로 S_bar를 수집하므로 sigma가 정상 범위(0.03~0.10)에 들어온다.

    Args:
        vit:          ViTWithAdaMerge().vit  (순수 ViT backbone)
        prev_stats:   이전 iteration stats {layer_idx: {"mu": float, "sigma": float}}
        data_dir:     ImageNet 데이터 디렉터리
        subset_ratio: 훈련 데이터 사용 비율
        r_max:        최대 병합 수
        alpha:        sigmoid 민감도
        num_layers:   ViT 레이어 수
        device:       "cuda" or "cpu"
        seed:         랜덤 시드

    Returns:
        stats: {layer_idx: {"mu": float, "sigma": float}}
    """
    # 훈련 데이터 서브셋 로드
    train_table = load_arrow(data_dir, "train")
    total   = len(train_table)
    n_sub   = int(total * subset_ratio)
    rng     = np.random.RandomState(seed)
    idxs    = rng.choice(total, n_sub, replace=False)

    dataset = ArrowDataset(train_table, idxs.tolist(), TRANSFORM)
    loader  = DataLoader(dataset, batch_size=64, shuffle=False,
                         num_workers=4, pin_memory=True)

    # 이미지 단위로 수집 (precompute_stats.py 방식)
    collector = {l: [] for l in range(num_layers)}

    vit.eval()

    for imgs, _ in tqdm(loader, desc="  plain-cosine 추출", leave=False):
        imgs = imgs.to(device)
        B = imgs.shape[0]

        x = vit.patch_embed(imgs)
        cls = vit.cls_token.expand(B, -1, -1)
        x   = torch.cat([cls, x], dim=1)
        x   = vit.pos_drop(x + vit.pos_embed)

        for l, block in enumerate(vit.blocks):
            patch  = x[:, 1:]           # CLS 제외
            T      = patch.shape[1]
            T_half = T // 2

            x_A = patch[:, :T_half]
            x_B = patch[:, T_half:]

            # Plain cosine similarity
            xA_n = F.normalize(x_A, dim=-1)
            xB_n = F.normalize(x_B, dim=-1)
            sim  = torch.bmm(xA_n, xB_n.transpose(1, 2))  # [B, T/2, T/2]

            # ── 이미지 단위로 S_bar 수집 (sigma 정확도 보장) ──
            max_sim = sim.amax(dim=-1)   # [B, T/2]
            for n in range(B):
                collector[l].append(max_sim[n].mean().item())

            # ── adaptive r 결정 (배치 평균 S_bar 기준) ──
            mu_l    = prev_stats[l]["mu"]
            sigma_l = prev_stats[l]["sigma"]
            r = compute_adaptive_r(sim, mu_l, sigma_l, r_max, alpha)

            # ── ToMe 방식 병합 ──
            patch = _tome_merge_batch(patch, sim, r)
            x = torch.cat([x[:, :1], patch], dim=1)
            x = block(x)

    # 통계 계산 및 출력
    stats = {}
    print(f"\n  Layer | mu     | sigma")
    print(f"  ------|--------|-------")
    for l in range(num_layers):
        vals  = torch.tensor(collector[l])
        mu    = vals.mean().item()
        sigma = max(vals.std().item(), 1e-4)
        stats[l] = {"mu": mu, "sigma": sigma}
        print(f"  l={l:2d}  | {mu:.4f} | {sigma:.4f}")

    return stats


# ─────────────────────────────────────────
# 평가 + 토큰 수 측정
# ─────────────────────────────────────────

@torch.no_grad()
def evaluate(model, loader, device, desc="Evaluating") -> float:
    model.eval()
    correct = total = 0
    for imgs, labels in tqdm(loader, desc=desc, leave=False):
        imgs, labels = imgs.to(device), labels.to(device)
        with autocast():
            logits = model(imgs)
        correct += (logits.argmax(-1) == labels).sum().item()
        total   += labels.size(0)
    return correct / total * 100


@torch.no_grad()
def evaluate_with_token_count(model, loader, device, desc, n_layers=12):
    """정확도 + 레이어별 평균 병합 토큰 수 측정."""
    model.eval()

    # ── 토큰 수 측정 (n_measure개 이미지) ──
    n_measure = 500
    total_merged_per_layer = [0.0] * n_layers
    n_measured = 0

    measure_iter = iter(loader)
    imgs_list = []
    while len(imgs_list) * loader.batch_size < n_measure:
        try:
            imgs, _ = next(measure_iter)
            imgs_list.append(imgs)
        except StopIteration:
            break

    for imgs_batch in imgs_list:
        imgs_batch = imgs_batch.to(device)
        B = imgs_batch.shape[0]

        x = model.vit.patch_embed(imgs_batch)
        cls = model.vit.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = model.vit.pos_drop(x + model.vit.pos_embed)

        for l, block in enumerate(model.vit.blocks):
            T_before = x.shape[1]
            x = model.ada_merge(x, layer_idx=l)
            T_after  = x.shape[1]
            total_merged_per_layer[l] += (T_before - T_after) * B
            x = block(x)

        n_measured += B

    avg_merged_per_layer = [
        total_merged_per_layer[l] / n_measured for l in range(n_layers)
    ]
    total_avg_merged = sum(avg_merged_per_layer)

    # ── 전체 정확도 평가 ──
    acc = evaluate(model, loader, device, desc=desc)

    return acc, avg_merged_per_layer, total_avg_merged


# ─────────────────────────────────────────
# Component-wise AdaMerge 변형
# ─────────────────────────────────────────

class AdaMergeVariant(AdaMerge):
    def __init__(self, use_salience, use_adaptive, fixed_r, **kwargs):
        super().__init__(**kwargs)
        self.use_salience = use_salience
        self.use_adaptive = use_adaptive
        self.fixed_r      = fixed_r

    def forward(self, x: torch.Tensor, layer_idx: int) -> torch.Tensor:
        cls_t   = x[:, :1, :]
        patches = x[:, 1:, :]
        T_half  = patches.shape[1] // 2

        x_A = patches[:, :T_half]
        x_B = patches[:, T_half:]

        # ── 유사도 계산 ──
        if self.use_salience:
            S   = compute_token_salience(patches)
            sim = compute_salience_weighted_similarity(x_A, x_B, S)
        else:
            S   = None
            xA_n = F.normalize(x_A, dim=-1)
            xB_n = F.normalize(x_B, dim=-1)
            sim  = torch.bmm(xA_n, xB_n.transpose(1, 2))

        # ── r 결정 ──
        if self.use_adaptive:
            mu_l    = self.stats[layer_idx]["mu"]
            sigma_l = self.stats[layer_idx]["sigma"]
            r = compute_adaptive_r(sim, mu_l, sigma_l, self.r_max, self.alpha)
        else:
            r = self.fixed_r

        r = min(r, T_half)

        # ── 병합 ──
        if self.use_salience:
            merged, _ = bipartite_merge(patches, S, r, sim)
        else:
            merged = _tome_merge_batch(patches, sim, r)

        return torch.cat([cls_t, merged], dim=1)


# ─────────────────────────────────────────
# Main
# ─────────────────────────────────────────

def main(args):
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[장치] {device}")
    print(f"r_max={args.r_max}, tome_r={args.tome_r}")

    # salience-weighted stats (Full AdaMerge / Salience only용)
    sw_stats = load_stats(args.stats_path)

    # Val loader
    print("\n[데이터] Val set 로드 중...")
    val_table  = load_arrow(args.data_dir, "validation")
    val_ds     = ArrowDataset(val_table, transform=TRANSFORM)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                            shuffle=False, num_workers=4, pin_memory=True)
    print(f"  Val: {len(val_ds)}장")

    # ── Step 1: plain-cosine stats 추출 (Adaptive r only 전용) ──
    # precompute_stats.py 방식 2패스 수렴, monkey-patch 없음
    print("\n[Step 1] plain-cosine stats 추출 (2패스)")

    # 기본 ViT 로드 (stats extraction용)
    base_model = ViTWithAdaMerge(
        r_max=args.r_max, alpha=args.alpha,
        stats_path=args.stats_path, pretrained=True,
    ).to(device)
    vit = base_model.vit

    default_stats = {l: {"mu": 0.5, "sigma": 0.1} for l in range(12)}

    print("\n  [1차]")
    plain_stats_1st = extract_plain_cosine_stats_clean(
        vit, default_stats,
        args.data_dir, args.subset_ratio,
        args.r_max, args.alpha,
        device=device,
    )

    print("\n  [2차]")
    plain_stats = extract_plain_cosine_stats_clean(
        vit, plain_stats_1st,
        args.data_dir, args.subset_ratio,
        args.r_max, args.alpha,
        device=device,
    )
    print("\n[plain-cosine stats] 2차 수렴 완료")
    del base_model
    torch.cuda.empty_cache()

    # ── Step 2: 4가지 variant 평가 ──
    print("\n[Step 2] Component ablation 평가 시작")

    configs = [
        # (label,              use_sw,  use_adp, stats)
        ("ToMe (baseline)",    False,   False,   sw_stats),
        ("Salience only",      True,    False,   sw_stats),
        ("Adaptive r only",    False,   True,    plain_stats),
        ("Full AdaMerge",      True,    True,    sw_stats),
    ]

    results = {}
    for label, use_sw, use_adp, stats in configs:
        print(f"\n── {label} (SW={use_sw}, Adp={use_adp}) ──")

        model = ViTWithAdaMerge(
            r_max=args.r_max, alpha=args.alpha,
            stats_path=args.stats_path, pretrained=True,
        ).to(device)
        model.ada_merge = AdaMergeVariant(
            use_salience=use_sw,
            use_adaptive=use_adp,
            fixed_r=args.tome_r,
            num_layers=12,
            r_max=args.r_max,
            alpha=args.alpha,
            stats=stats,
        ).to(device)

        acc, avg_per_layer, total_merged = evaluate_with_token_count(
            model, val_loader, device, desc=label
        )
        print(f"  Top-1 Acc:     {acc:.2f}%")
        print(f"  Avg merged:    {total_merged:.1f} tokens/image")

        results[label] = {
            "use_salience":             use_sw,
            "use_adaptive":             use_adp,
            "top1_acc":                 round(acc, 2),
            "avg_merged_total":         round(total_merged, 2),
            "avg_merged_per_layer":     [round(v, 2) for v in avg_per_layer],
        }
        del model
        torch.cuda.empty_cache()

    # ── 결과 출력 ──
    print("\n=== Component Ablation Results ===")
    baseline_acc = results["ToMe (baseline)"]["top1_acc"]
    print(f"{'Config':30s} | {'SW':>5} | {'Adp':>5} | "
          f"{'Top-1':>8} | {'Δ vs ToMe':>10} | {'Merged':>8}")
    print("-" * 80)
    for label, r in results.items():
        delta  = r["top1_acc"] - baseline_acc
        sw_sym = "✓" if r["use_salience"] else "✗"
        adp_sym= "✓" if r["use_adaptive"] else "✗"
        print(f"{label:30s} | {sw_sym:>5} | {adp_sym:>5} | "
              f"{r['top1_acc']:>8.2f}% | {delta:>+10.2f}%p | "
              f"{r['avg_merged_total']:>8.1f}")

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[저장] {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",     type=str,   required=True)
    parser.add_argument("--stats_path",   type=str,   required=True)
    parser.add_argument("--r_max",        type=int,   default=17)
    parser.add_argument("--tome_r",       type=int,   default=6)
    parser.add_argument("--alpha",        type=float, default=3.0)
    parser.add_argument("--subset_ratio", type=float, default=0.01)
    parser.add_argument("--batch_size",   type=int,   default=128)
    parser.add_argument("--output",       type=str,
                        default="./ablation_results/component.json")
    args = parser.parse_args()
    main(args)