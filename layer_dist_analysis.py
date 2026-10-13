"""
layer_dist_analysis.py
-----------------------
레이어별 r_l 분포 분석 및 시각화.

ImageNet-1k validation set 전체(50K)에서
각 이미지의 레이어별 r_l을 수집하고
box plot + 통계 테이블 생성.

사용법:
  python layer_dist_analysis.py \
    --data_dir   ./data/imagenet1k \
    --stats_path ./stats_rmax12_augreg2.json \
    --r_max      12 \
    --output_dir ./layer_dist_results
"""

import os
import json
import argparse
import numpy as np
import torch
import glob
import io
import pyarrow as pa
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats as scipy_stats

from adamerge import AdaMerge, compute_salience_weighted_similarity, compute_adaptive_r
from salience import compute_token_salience
from vit_adamerge import ViTWithAdaMerge


# ─────────────────────────────────────────
# 데이터 로드 (Validation Arrow)
# ─────────────────────────────────────────

class ValArrowDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir: str, transform=None):
        arrow_dir = os.path.join(
            data_dir,
            "ILSVRC___imagenet-1k", "default", "0.0.0",
            "49e2ee26f3810fb5a7536bbf732a7b07389a47b5",
        )
        val_files = sorted(glob.glob(
            os.path.join(arrow_dir, "imagenet-1k-validation-*.arrow")
        ))
        if not val_files:
            raise FileNotFoundError(f"Validation arrow 파일 없음: {arrow_dir}")

        print(f"  Validation arrow 파일 {len(val_files)}개 로드 중...")
        tables = []
        for f in val_files:
            with open(f, "rb") as src:
                reader = pa.ipc.open_stream(src)
                tables.append(reader.read_all())
        self.table = pa.concat_tables(tables)
        self.transform = transform
        print(f"  총 {len(self.table)}장 로드 완료")

    def __len__(self):
        return len(self.table)

    def __getitem__(self, idx):
        row = self.table.slice(idx, 1)
        img_data = row.column("image")[0].as_py()
        if isinstance(img_data, dict):
            img = Image.open(io.BytesIO(img_data["bytes"])).convert("RGB")
        else:
            img = Image.open(io.BytesIO(img_data)).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, 0


# ─────────────────────────────────────────
# r_l 수집
# ─────────────────────────────────────────

def collect_rl_distribution(
    loader, model, num_layers: int, device
) -> dict:
    """
    각 이미지의 레이어별 r_l 수집.
    Returns: {layer_idx: [r_l per image]}
    """
    collector = {l: [] for l in range(num_layers)}

    model.eval()
    with torch.no_grad():
        for imgs, _ in tqdm(loader, desc="r_l 수집 중"):
            imgs = imgs.to(device)
            B = imgs.shape[0]

            x = model.vit.patch_embed(imgs)
            cls = model.vit.cls_token.expand(B, -1, -1)
            x = torch.cat([cls, x], dim=1)
            x = model.vit.pos_drop(x + model.vit.pos_embed)

            for l, block in enumerate(model.vit.blocks):
                # r_l 계산 (AdaMerge forward 내부 로직)
                cls_tok = x[:, :1, :]
                patches = x[:, 1:, :]
                T = patches.shape[1]
                T_half = T // 2

                S   = compute_token_salience(patches)
                x_A = patches[:, :T_half]
                x_B = patches[:, T_half:]
                sim = compute_salience_weighted_similarity(x_A, x_B, S)

                mu_l    = model.ada_merge.stats[l]["mu"]
                sigma_l = model.ada_merge.stats[l]["sigma"]
                r = compute_adaptive_r(
                    sim, mu_l, sigma_l,
                    model.ada_merge.r_max,
                    model.ada_merge.alpha
                )
                # 배치 내 모든 이미지에 동일한 r 적용 (batch-level scalar)
                for _ in range(B):
                    collector[l].append(r)

                # 실제 병합 진행
                x = model.ada_merge(x, layer_idx=l)
                x = block(x)

    return collector


# ─────────────────────────────────────────
# Figure 생성
# ─────────────────────────────────────────

def make_layer_dist_figure(
    collector: dict,
    r_max: int,
    output_path: str,
    tome_r: int = None,
):
    """
    Box plot: AdaMerge r_l distribution per layer
    + optional ToMe fixed r baseline
    """
    num_layers = len(collector)
    layers = list(range(num_layers))

    data = [collector[l] for l in layers]

    fig, ax = plt.subplots(figsize=(9, 4))

    # Box plot
    bp = ax.boxplot(
        data,
        positions=layers,
        widths=0.55,
        patch_artist=True,
        showfliers=False,
        medianprops=dict(color="#185FA5", linewidth=2),
        boxprops=dict(facecolor="#B5D4F4", alpha=0.8),
        whiskerprops=dict(linewidth=1.2),
        capprops=dict(linewidth=1.2),
    )

    # ToMe fixed r baseline
    if tome_r is not None:
        ax.axhline(
            y=tome_r, color="#A32D2D", linewidth=1.5,
            linestyle="--", label=f"ToMe fixed $r={tome_r}$"
        )

    # r_max ceiling
    ax.axhline(
        y=r_max, color="#5F5E5A", linewidth=1.0,
        linestyle=":", alpha=0.6, label=f"$r_{{\\max}}={r_max}$"
    )

    # Mean line
    means = [np.mean(collector[l]) for l in layers]
    ax.plot(layers, means, "D-", color="#185FA5",
            markersize=4, linewidth=1.2, label="Mean $r_l$")

    ax.set_xlabel("ViT Block $l$", fontsize=11)
    ax.set_ylabel("Merging count $r_l$", fontsize=11)
    ax.set_title(
        f"Per-block $r_l$ distribution over ImageNet-1k val"
        f" ($r_{{\\max}}={r_max}$)",
        fontsize=11
    )
    ax.set_xticks(layers)
    ax.set_xticklabels([str(l) for l in layers])
    ax.set_ylim(-0.5, r_max + 1.5)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, axis="y", alpha=0.3, linewidth=0.5)

    plt.tight_layout()
    plt.savefig(output_path, format="pdf", dpi=300, bbox_inches="tight")
    plt.savefig(output_path.replace(".pdf", ".png"),
                format="png", dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[Figure] 저장: {output_path}")


# ─────────────────────────────────────────
# 통계 계산 + 출력
# ─────────────────────────────────────────

def compute_and_print_stats(collector: dict) -> dict:
    print("\n=== Per-Block r_l Statistics ===")
    print(f"{'Block':>6} | {'E[r_l]':>8} | {'Std[r_l]':>10} | "
          f"{'Skewness':>10} | {'Min':>5} | {'Max':>5}")
    print("-" * 60)

    result = {}
    for l, vals in collector.items():
        arr  = np.array(vals, dtype=float)
        mean = arr.mean()
        std  = arr.std()
        skew = scipy_stats.skew(arr)
        vmin = arr.min()
        vmax = arr.max()
        result[l] = {"mean": mean, "std": std, "skew": skew,
                     "min": vmin, "max": vmax}
        print(f"{l:>6} | {mean:>8.3f} | {std:>10.3f} | "
              f"{skew:>10.3f} | {vmin:>5.0f} | {vmax:>5.0f}")

    return result


# ─────────────────────────────────────────
# Main
# ─────────────────────────────────────────

def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[장치] {device}")

    transform = transforms.Compose([
        transforms.Resize(249, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    print("[데이터] Validation set 로드 중...")
    dataset = ValArrowDataset(args.data_dir, transform=transform)
    loader  = DataLoader(
        dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=4, pin_memory=True
    )

    print(f"[모델] ViTWithAdaMerge 로드 중 (r_max={args.r_max})...")
    model = ViTWithAdaMerge(
        r_max=args.r_max,
        alpha=args.alpha,
        stats_path=args.stats_path,
        pretrained=True,
    ).to(device)
    model.eval()

    # r_l 수집
    collector = collect_rl_distribution(loader, model, 12, device)

    # 통계 계산
    stats = compute_and_print_stats(collector)

    # JSON 저장
    out = {str(l): v for l, v in stats.items()}
    with open(os.path.join(args.output_dir, "layer_dist_stats.json"), "w") as f:
        json.dump(out, f, indent=2)

    # Figure 생성
    fig_path = os.path.join(args.output_dir, "layer_dist.pdf")
    make_layer_dist_figure(
        collector, args.r_max, fig_path,
        tome_r=args.tome_r
    )
    print(f"\n[완료] 결과: {args.output_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",    type=str,   default="./data/imagenet1k")
    parser.add_argument("--stats_path",  type=str,   required=True)
    parser.add_argument("--r_max",       type=int,   default=12)
    parser.add_argument("--alpha",       type=float, default=3.0)
    parser.add_argument("--tome_r",      type=int,   default=4,
                        help="ToMe 비교 baseline r (FLOPs-matched)")
    parser.add_argument("--batch_size",  type=int,   default=128)
    parser.add_argument("--output_dir",  type=str,   default="./layer_dist_results")
    args = parser.parse_args()
    main(args)