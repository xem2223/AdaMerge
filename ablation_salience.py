"""
ablation_salience.py
---------------------
Alternative Salience Definitions 비교 실험.

비교 대상:
  1. Outgoing affinity (Ours)
       s_i = Σ_j Â_ji  (softmax(A)의 column-wise sum)
  2. Attention entropy
       s_i = -Σ_j Â_ij * log(Â_ij + ε)  (row-wise entropy)
       → 낮을수록 특정 토큰에 집중 → 중요한 토큰
       → 높을수록 분산 주의 → 덜 중요한 토큰
       (음수 취해서 entropy가 낮은 토큰이 높은 salience 갖도록)
  3. [CLS]-attention
       s_i = cos(x_cls, x_i)  (CLS 토큰과 각 패치의 코사인 유사도)

모두 동일한 r (fixed, ToMe 방식)로 평가 → salience 정의 효과만 격리.

사용법:
  python ablation_salience.py \
    --data_dir   ./data/imagenet1k \
    --r_max      17 \
    --tome_r     6 \
    --batch_size 128 \
    --output     ./ablation_results/salience_defs.json
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

from adamerge import AdaMerge, bipartite_merge, compute_adaptive_r
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
# Salience 정의
# ─────────────────────────────────────────

def salience_incoming(patches: torch.Tensor) -> torch.Tensor:
    """
    Ours (AdaMerge): 열 방향 합 (Incoming affinity).
    A = X X^T, Â = softmax(A, dim=-1)
    s_j = Σ_i Â_ij  ← 다른 토큰들이 j에게 주목하는 정도 (incoming)
    → 다른 토큰들로부터 많이 참조되는 토큰이 높은 salience.
    """
    A     = torch.bmm(patches, patches.transpose(1, 2))   # [B, N, N]
    A_hat = torch.softmax(A, dim=-1)                       # [B, N, N]
    s     = A_hat.sum(dim=1)                               # [B, N] 열 방향 합 (incoming)
    s_min = s.amin(dim=-1, keepdim=True)
    s_max = s.amax(dim=-1, keepdim=True)
    return (s - s_min) / (s_max - s_min + 1e-6)


def salience_outgoing(patches: torch.Tensor) -> torch.Tensor:
    """
    대안: 행 방향 합 (Outgoing affinity).
    s_i = Σ_j Â_ij  ← token i가 다른 토큰들에게 주목하는 정도 (outgoing)
    → 자신이 다른 토큰들을 많이 참조하는 토큰이 높은 salience.
    """
    A     = torch.bmm(patches, patches.transpose(1, 2))   # [B, N, N]
    A_hat = torch.softmax(A, dim=-1)                       # [B, N, N]
    s     = A_hat.sum(dim=2)                               # [B, N] 행 방향 합 (outgoing)
    s_min = s.amin(dim=-1, keepdim=True)
    s_max = s.amax(dim=-1, keepdim=True)
    return (s - s_min) / (s_max - s_min + 1e-6)


def salience_entropy(patches: torch.Tensor) -> torch.Tensor:
    """
    Attention entropy: s_i = -Σ_j Â_ij * log(Â_ij + ε)
    → 낮은 entropy = 집중된 attention → 중요한 토큰.
    음수 entropy를 salience로 사용 (낮은 entropy가 높은 salience).
    """
    A     = torch.bmm(patches, patches.transpose(1, 2))
    A_hat = torch.softmax(A, dim=-1)
    eps   = 1e-8
    entropy = -(A_hat * (A_hat + eps).log()).sum(dim=-1)   # [B, N] row-wise
    s     = -entropy                                        # 낮은 entropy → 높은 salience
    s_min = s.amin(dim=-1, keepdim=True)
    s_max = s.amax(dim=-1, keepdim=True)
    return (s - s_min) / (s_max - s_min + 1e-6)


def salience_cls(cls_token: torch.Tensor, patches: torch.Tensor) -> torch.Tensor:
    """
    [CLS]-attention proxy: CLS 토큰과 각 패치의 코사인 유사도.
    s_i = cos(x_cls, x_i)
    → CLS 토큰이 많이 참조하는 패치가 높은 salience.
    """
    cls_n  = F.normalize(cls_token, dim=-1)                # [B, 1, d]
    pat_n  = F.normalize(patches,   dim=-1)                # [B, N, d]
    s      = (cls_n * pat_n).sum(dim=-1)                   # [B, N]
    s_min  = s.amin(dim=-1, keepdim=True)
    s_max  = s.amax(dim=-1, keepdim=True)
    return (s - s_min) / (s_max - s_min + 1e-6)


# ─────────────────────────────────────────
# Salience-variant AdaMerge
# ─────────────────────────────────────────

class AdaMergeSalienceVariant(AdaMerge):
    """
    salience_type: 'outgoing' | 'entropy' | 'cls' | 'tome'
    fixed_r:       use_adaptive=False 일 때 고정 r
    use_adaptive:  True → adaptive r 사용 / False → fixed r
    """

    def __init__(self, salience_type: str, fixed_r: int,
                 use_adaptive: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.salience_type = salience_type
        self.fixed_r       = fixed_r
        self.use_adaptive  = use_adaptive

    def forward(self, x: torch.Tensor, layer_idx: int) -> torch.Tensor:
        cls_t   = x[:, :1, :]
        patches = x[:, 1:, :]
        T_half  = patches.shape[1] // 2

        if self.salience_type == "tome":
            xA_n = F.normalize(patches[:, :T_half], dim=-1)
            xB_n = F.normalize(patches[:, T_half:], dim=-1)
            sim  = torch.bmm(xA_n, xB_n.transpose(1, 2))
        else:
            if self.salience_type == "incoming":
                S = salience_incoming(patches)
            elif self.salience_type == "outgoing":
                S = salience_outgoing(patches)
            elif self.salience_type == "entropy":
                S = salience_entropy(patches)
            elif self.salience_type == "cls":
                S = salience_cls(cls_t, patches)

            xA_n = F.normalize(patches[:, :T_half], dim=-1)
            xB_n = F.normalize(patches[:, T_half:], dim=-1)
            cos  = torch.bmm(xA_n, xB_n.transpose(1, 2))
            sim  = S[:, :T_half].unsqueeze(-1) * cos

        # r 결정
        if self.use_adaptive:
            mu_l    = self.stats[layer_idx]["mu"]
            sigma_l = self.stats[layer_idx]["sigma"]
            r = compute_adaptive_r(sim, mu_l, sigma_l, self.r_max, self.alpha)
        else:
            r = self.fixed_r

        r = min(r, T_half)

        if self.salience_type == "tome":
            merged = self._unweighted_merge(patches, sim, r)
        else:
            merged = self._salience_weighted_merge(patches, S, sim, r)

        return torch.cat([cls_t, merged], dim=1)

    def _unweighted_merge(self, patches, sim, r):
        """ToMe 스타일 단순 평균 병합."""
        B, T, D = patches.shape
        T_half  = T // 2
        r       = min(r, T_half)
        if r == 0:
            return patches
        x_A, x_B = patches[:, :T_half], patches[:, T_half:]
        _, top_i  = sim.amax(dim=-1).topk(r, dim=-1)
        j_idx     = sim.argmax(dim=-1).gather(1, top_i)
        xA_top    = x_A.gather(1, top_i.unsqueeze(-1).expand(-1,-1,D))
        xB_tgt    = x_B.gather(1, j_idx.unsqueeze(-1).expand(-1,-1,D))
        x_B       = x_B.clone()
        x_B.scatter_(1, j_idx.unsqueeze(-1).expand(-1,-1,D),
                     (xA_top + xB_tgt) / 2.0)
        mask = torch.zeros(B, T_half, dtype=torch.bool, device=patches.device)
        mask.scatter_(1, top_i, True)
        x_A = x_A[~mask].view(B, T_half - r, D)
        return torch.cat([x_A, x_B], dim=1)

    def _salience_weighted_merge(self, patches, S, sim, r):
        """Salience 가중 평균 병합."""
        B, T, D = patches.shape
        T_half  = T // 2
        r       = min(r, T_half)
        if r == 0:
            return patches
        x_A, x_B = patches[:, :T_half], patches[:, T_half:]
        S_A,  S_B = S[:, :T_half],      S[:, T_half:]

        _, top_i  = sim.amax(dim=-1).topk(r, dim=-1)
        j_idx     = sim.argmax(dim=-1).gather(1, top_i)

        xA_top = x_A.gather(1, top_i.unsqueeze(-1).expand(-1,-1,D))
        xB_tgt = x_B.gather(1, j_idx.unsqueeze(-1).expand(-1,-1,D))
        sA_top = S_A.gather(1, top_i).unsqueeze(-1)
        sB_tgt = S_B.gather(1, j_idx).unsqueeze(-1)
        w      = sA_top / (sA_top + sB_tgt + 1e-8)
        merged = w * xA_top + (1 - w) * xB_tgt

        x_B = x_B.clone()
        x_B.scatter_(1, j_idx.unsqueeze(-1).expand(-1,-1,D), merged)
        mask = torch.zeros(B, T_half, dtype=torch.bool, device=patches.device)
        mask.scatter_(1, top_i, True)
        x_A = x_A[~mask].view(B, T_half - r, D)
        return torch.cat([x_A, x_B], dim=1)


# ─────────────────────────────────────────
# 평가
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
def extract_salience_stats(
    model, data_dir, subset_ratio, salience_type,
    num_layers, device, r_max, alpha,
    seed=42, prev_stats=None
):
    """
    각 salience 정의 기준으로 (mu_l, sigma_l) 추출.
    prev_stats=None  → 1차: fixed r=4로 병합하며 S_bar 수집
    prev_stats=dict  → 2차: prev_stats 기반 adaptive r로 병합 (self-consistent)
    """
    pass_label = "2차 (수렴)" if prev_stats is not None else "1차"
    print(f"\n  [{salience_type}] {pass_label} stats 추출 중...")
    train_table = load_arrow(data_dir, "train")
    total = len(train_table)
    n_sub = int(total * subset_ratio)
    rng   = np.random.RandomState(seed)
    idxs  = rng.choice(total, n_sub, replace=False)

    dataset = ArrowDataset(train_table, idxs.tolist(), TRANSFORM)
    loader  = DataLoader(dataset, batch_size=64, shuffle=False,
                         num_workers=4, pin_memory=True)

    collector = {l: [] for l in range(num_layers)}
    model.eval()

    for imgs, _ in tqdm(loader, desc=f"    {salience_type} {pass_label}", leave=False):
        imgs = imgs.to(device)
        B    = imgs.shape[0]

        x = model.vit.patch_embed(imgs)
        cls = model.vit.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = model.vit.pos_drop(x + model.vit.pos_embed)

        for l, block in enumerate(model.vit.blocks):
            cls_t   = x[:, :1, :]
            patches = x[:, 1:, :]
            T_half  = patches.shape[1] // 2
            x_A     = patches[:, :T_half]
            x_B     = patches[:, T_half:]

            # salience 종류별 sim 계산
            if salience_type == "tome":
                xA_n = F.normalize(x_A, dim=-1)
                xB_n = F.normalize(x_B, dim=-1)
                sim  = torch.bmm(xA_n, xB_n.transpose(1, 2))
            elif salience_type in ("incoming", "outgoing", "entropy", "cls"):
                if salience_type == "incoming":
                    S = salience_incoming(patches)
                elif salience_type == "outgoing":
                    S = salience_outgoing(patches)
                elif salience_type == "entropy":
                    S = salience_entropy(patches)
                elif salience_type == "cls":
                    S = salience_cls(cls_t, patches)
                xA_n = F.normalize(x_A, dim=-1)
                xB_n = F.normalize(x_B, dim=-1)
                cos  = torch.bmm(xA_n, xB_n.transpose(1, 2))
                sim  = S[:, :T_half].unsqueeze(-1) * cos
            else:
                raise ValueError(f"Unknown salience_type: {salience_type}")

            S_bar = sim.amax(dim=-1).mean().item()
            collector[l].append(S_bar)

            # r 결정: 1차=fixed r=4, 2차=adaptive r (prev_stats 기반)
            if prev_stats is None:
                r = min(4, T_half)
            else:
                import math
                mu_l    = prev_stats[l]["mu"]
                sigma_l = prev_stats[l]["sigma"]
                z   = (S_bar - mu_l) / max(sigma_l, 1e-4)
                sig = torch.sigmoid(torch.tensor(alpha * z)).item()
                r   = min(max(math.floor(r_max * sig), 0), T_half)

            if r > 0:
                _, top_i = sim.amax(dim=-1).topk(r, dim=-1)
                j_idx    = sim.argmax(dim=-1).gather(1, top_i)
                D        = patches.shape[-1]
                xA_top   = x_A.gather(1, top_i.unsqueeze(-1).expand(-1,-1,D))
                xB_tgt   = x_B.gather(1, j_idx.unsqueeze(-1).expand(-1,-1,D))
                x_B2     = x_B.clone()
                x_B2.scatter_(1, j_idx.unsqueeze(-1).expand(-1,-1,D),
                              (xA_top + xB_tgt) / 2.0)
                mask = torch.zeros(B, T_half, dtype=torch.bool, device=device)
                mask.scatter_(1, top_i, True)
                x_A2     = x_A[~mask].view(B, T_half - r, D)
                patches  = torch.cat([x_A2, x_B2], dim=1)

            x = block(torch.cat([cls_t, patches], dim=1))

    stats = {}
    for l in range(num_layers):
        vals = torch.tensor(collector[l], dtype=torch.float32)
        stats[l] = {
            "mu":    vals.mean().item(),
            "sigma": max(vals.std().item(), 1e-4),
        }
        print(f"    l={l:2d}  mu={stats[l]['mu']:.4f}  "
              f"sigma={stats[l]['sigma']:.4f}")
    return stats


# ─────────────────────────────────────────
# Main
# ─────────────────────────────────────────

def main(args):
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[장치] {device}")
    print(f"mode={args.mode}, tome_r={args.tome_r}, r_max={args.r_max}")

    # Val loader
    print("\n[데이터] Val set 로드 중...")
    val_table  = load_arrow(args.data_dir, "validation")
    val_ds     = ArrowDataset(val_table, transform=TRANSFORM)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                            shuffle=False, num_workers=4, pin_memory=True)
    print(f"  Val: {len(val_ds)}장")

    sal_types = ["tome", "incoming", "outgoing", "entropy", "cls"]
    labels    = [
        "ToMe (baseline)",
        "Incoming affinity (Ours)",
        "Outgoing affinity",
        "Attention entropy",
        "[CLS]-attention",
    ]

    # Adaptive r 모드: salience별 stats 먼저 추출
    per_stats = {}
    if args.mode == "adaptive_r":
        print("\n[Stats 추출] salience별 stats 추출 중...")
        base_model = ViTWithAdaMerge(
            r_max=args.r_max, alpha=3.0,
            stats_path=None, pretrained=True,
        ).to(device)
        for sal_type in sal_types:
            # 1차 추출
            stats_1st = extract_salience_stats(
                base_model, args.data_dir, args.subset_ratio,
                sal_type, 12, device,
                r_max=args.r_max, alpha=3.0,
                prev_stats=None,
            )
            # 2차 수렴 (prev_stats 기반 adaptive r 적용)
            stats_2nd = extract_salience_stats(
                base_model, args.data_dir, args.subset_ratio,
                sal_type, 12, device,
                r_max=args.r_max, alpha=3.0,
                prev_stats=stats_1st,
            )
            per_stats[sal_type] = stats_2nd
            print(f"  [{sal_type}] 수렴 완료")
        del base_model
        torch.cuda.empty_cache()

    # 평가
    results = {}
    for sal_type, label in zip(sal_types, labels):
        print(f"\n── {label} ──")

        if args.mode == "fixed_r":
            stats   = {l: {"mu": 0.5, "sigma": 0.1} for l in range(12)}
            fixed_r = args.tome_r
            use_adp = False
        else:
            stats   = per_stats[sal_type]
            fixed_r = args.tome_r
            use_adp = True

        model = ViTWithAdaMerge(
            r_max=args.r_max, alpha=3.0,
            stats_path=None, pretrained=True,
        ).to(device)
        model.ada_merge = AdaMergeSalienceVariant(
            salience_type=sal_type,
            fixed_r=fixed_r,
            use_adaptive=use_adp,
            num_layers=12,
            r_max=args.r_max,
            alpha=3.0,
            stats=stats,
        ).to(device)

        acc = evaluate(model, val_loader, device, desc=label)
        print(f"  Top-1 Acc: {acc:.2f}%")
        results[label] = {
            "salience_type": sal_type,
            "top1_acc":      round(acc, 2),
        }
        del model
        torch.cuda.empty_cache()

    # 결과 출력
    print(f"\n=== Alternative Salience Results (mode={args.mode}) ===")
    baseline = results["ToMe (baseline)"]["top1_acc"]
    print(f"{'Variant':35s} | {'Top-1':>8} | {'Δ vs ToMe':>10}")
    print("-" * 60)
    for label, r in results.items():
        delta = r["top1_acc"] - baseline
        print(f"{label:35s} | {r['top1_acc']:>8.2f}% | {delta:>+10.2f}%p")

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[저장] {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",     type=str, required=True)
    parser.add_argument("--r_max",        type=int, default=17)
    parser.add_argument("--tome_r",       type=int, default=6)
    parser.add_argument("--mode",         type=str, default="adaptive_r",
                        choices=["fixed_r", "adaptive_r"],
                        help="fixed_r: salience 정의만 비교 / "
                             "adaptive_r: adaptive r도 함께 켬")
    parser.add_argument("--subset_ratio", type=float, default=0.01)
    parser.add_argument("--batch_size",   type=int, default=128)
    parser.add_argument("--output",       type=str,
                        default="./ablation_results/salience_defs.json")
    args = parser.parse_args()
    main(args)