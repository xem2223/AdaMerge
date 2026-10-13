"""
merging_pattern_viz.py
-----------------------
AdaMerge vs ToMe 병합 패턴 비교 시각화.

Figure 구성:
  3행 × 3열: Original | AdaMerge survived | ToMe survived

사용법:
  python merging_pattern_viz.py \
    --data_dir   ./data/imagenet1k \
    --stats_path ./stats_rmax16_augreg2.json \
    --r_max      16 \
    --tome_r     5 \
    --output     figure_merging_pattern.pdf \
    --seed       42
"""

import os
import argparse
import random
import io
import glob
import numpy as np
import torch
import torch.nn.functional as F
import pyarrow as pa
from PIL import Image
from torchvision import transforms
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap

from salience import compute_token_salience
from adamerge import (
    compute_salience_weighted_similarity,
    compute_adaptive_r,
    bipartite_merge,
)
from vit_adamerge import ViTWithAdaMerge


# ─────────────────────────────────────────
# 설정
# ─────────────────────────────────────────
N_IMAGES   = 3
N_PATCHES  = 14
IMG_SIZE   = 224
MEAN = STD = [0.5, 0.5, 0.5]

MERGE_CMAP = LinearSegmentedColormap.from_list(
    "merge", ["#ff4444", "#44cc44"]
)


# ─────────────────────────────────────────
# 데이터 로드
# ─────────────────────────────────────────

def load_val_images(data_dir: str, n: int, seed: int = 42,
                    positions: list = None):
    """
    positions: seed로 샘플링한 n장 중 사용할 위치 리스트.
               예) [0, 1, 4] → 첫 번째, 두 번째, 다섯 번째 이미지만 반환.
               None이면 전체 n장 반환.
    """
    arrow_dir = os.path.join(
        data_dir,
        "ILSVRC___imagenet-1k", "default", "0.0.0",
        "49e2ee26f3810fb5a7536bbf732a7b07389a47b5",
    )
    val_files = sorted(glob.glob(
        os.path.join(arrow_dir, "imagenet-1k-validation-*.arrow")
    ))
    tables = []
    for f in val_files[:3]:
        with open(f, "rb") as src:
            reader = pa.ipc.open_stream(src)
            tables.append(reader.read_all())
    table = pa.concat_tables(tables)

    random.seed(seed)
    all_indices = random.sample(range(len(table)), n)

    # positions 필터링
    if positions is not None:
        selected = [all_indices[p] for p in positions]
    else:
        selected = all_indices

    vis_tf = transforms.Compose([
        transforms.Resize(249, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
    ])
    ten_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])

    imgs_vis, imgs_tensor = [], []
    for idx in selected:
        row = table.slice(idx, 1)
        raw = row.column("image")[0].as_py()
        img = Image.open(io.BytesIO(
            raw["bytes"] if isinstance(raw, dict) else raw
        )).convert("RGB")
        vis = vis_tf(img)
        imgs_vis.append(vis)
        imgs_tensor.append(ten_tf(vis).unsqueeze(0))
        print(f"  idx={idx}, size={img.size}")
    return imgs_vis, imgs_tensor


# ─────────────────────────────────────────
# AdaMerge survived token map
# ─────────────────────────────────────────

def get_adamerge_survived(model, img_tensor, device):
    x = img_tensor.to(device)
    with torch.no_grad():
        x_emb = model.vit.patch_embed(x)
        cls = model.vit.cls_token.expand(1, -1, -1)
        x_emb = torch.cat([cls, x_emb], dim=1)
        x_emb = model.vit.pos_drop(x_emb + model.vit.pos_embed)

        indices = torch.arange(196, device=device).unsqueeze(0)

        for l, block in enumerate(model.vit.blocks):
            cls_t   = x_emb[:, :1, :]
            patches = x_emb[:, 1:, :]
            T_half  = patches.shape[1] // 2

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
            r = min(r, T_half)

            if r > 0:
                top_sim = sim.amax(dim=-1)
                _, top_idx = top_sim.topk(r, dim=-1)
                merge_mask = torch.zeros(1, T_half, dtype=torch.bool, device=device)
                merge_mask.scatter_(1, top_idx, True)

                idx_A      = indices[:, :T_half]
                idx_B      = indices[:, T_half:]
                idx_A_keep = idx_A[~merge_mask].unsqueeze(0)
                indices    = torch.cat([idx_A_keep, idx_B], dim=1)

                patches, S = bipartite_merge(patches, S, r, sim)

            x_emb = torch.cat([cls_t, patches], dim=1)
            x_emb = block(x_emb)

    survived = np.zeros(196, dtype=bool)
    survived[indices[0].cpu().numpy()] = True
    return survived


# ─────────────────────────────────────────
# ToMe survived token map
# ─────────────────────────────────────────

def get_tome_survived(model, img_tensor, device, tome_r: int):
    """ToMe fixed-r 병합으로 살아남은 토큰 추적."""
    x = img_tensor.to(device)
    with torch.no_grad():
        x_emb = model.vit.patch_embed(x)
        cls = model.vit.cls_token.expand(1, -1, -1)
        x_emb = torch.cat([cls, x_emb], dim=1)
        x_emb = model.vit.pos_drop(x_emb + model.vit.pos_embed)

        indices = torch.arange(196, device=device).unsqueeze(0)

        for l, block in enumerate(model.vit.blocks):
            cls_t   = x_emb[:, :1, :]
            patches = x_emb[:, 1:, :]
            T       = patches.shape[1]
            T_half  = T // 2
            r       = min(tome_r, T_half)

            if r > 0:
                # ToMe: cosine similarity between A and B groups
                x_A = patches[:, :T_half]
                x_B = patches[:, T_half:]
                xA_n = F.normalize(x_A, dim=-1)
                xB_n = F.normalize(x_B, dim=-1)
                sim  = torch.bmm(xA_n, xB_n.transpose(1, 2))  # [1, T/2, T/2]

                # Each A token matches best B token
                best_sim = sim.amax(dim=-1)                     # [1, T/2]
                _, top_idx = best_sim.topk(r, dim=-1)           # [1, r]

                merge_mask = torch.zeros(1, T_half, dtype=torch.bool, device=device)
                merge_mask.scatter_(1, top_idx, True)

                # Index tracking
                idx_A      = indices[:, :T_half]
                idx_B      = indices[:, T_half:]
                idx_A_keep = idx_A[~merge_mask].unsqueeze(0)
                indices    = torch.cat([idx_A_keep, idx_B], dim=1)

                # Token merging (unweighted average — ToMe style)
                best_j  = sim.argmax(dim=-1)            # [1, T/2]
                j_idx   = best_j.gather(1, top_idx)     # [1, r]

                xA_top  = x_A.gather(1, top_idx.unsqueeze(-1).expand(-1, -1, x_A.shape[-1]))
                xB_tgt  = x_B.gather(1, j_idx.unsqueeze(-1).expand(-1, -1, x_B.shape[-1]))
                merged  = (xA_top + xB_tgt) / 2.0

                x_B = x_B.clone()
                x_B.scatter_(1, j_idx.unsqueeze(-1).expand(-1, -1, x_B.shape[-1]), merged)

                keep = ~merge_mask
                x_A  = x_A[keep].unsqueeze(0).view(1, T_half - r, -1)
                patches = torch.cat([x_A, x_B], dim=1)

            x_emb = torch.cat([cls_t, patches], dim=1)
            x_emb = block(x_emb)

    survived = np.zeros(196, dtype=bool)
    survived[indices[0].cpu().numpy()] = True
    return survived


# ─────────────────────────────────────────
# 시각화 헬퍼
# ─────────────────────────────────────────

def survived_overlay(ax, img_vis, survived: np.ndarray, alpha_img=0.45):
    hm = survived.reshape(N_PATCHES, N_PATCHES).astype(float)
    hm_large = np.array(
        Image.fromarray((hm * 255).astype(np.uint8)).resize(
            (IMG_SIZE, IMG_SIZE), Image.NEAREST
        )
    ) / 255.0

    ax.imshow(img_vis, alpha=alpha_img)
    ax.imshow(hm_large, cmap=MERGE_CMAP, alpha=0.65, vmin=0, vmax=1)

    # 그리드
    patch_px = IMG_SIZE / N_PATCHES
    for i in range(N_PATCHES + 1):
        p = i * patch_px - 0.5
        ax.axhline(p, color="white", linewidth=0.3, alpha=0.5)
        ax.axvline(p, color="white", linewidth=0.3, alpha=0.5)

    n_surv = int(survived.sum())
    ax.text(0.5, 0.02, f"{n_surv}/196",
            ha="center", va="bottom", transform=ax.transAxes,
            fontsize=8, color="white",
            bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.5))


# ─────────────────────────────────────────
# Figure 생성
# ─────────────────────────────────────────

def make_figure(imgs_vis, ada_survived_list, tome_survived_list,
                r_max, tome_r, output_path):
    n_rows = N_IMAGES
    n_cols = 3  # Original | AdaMerge | ToMe

    col_labels = [
        "Original",
        f"AdaMerge ($r_{{\\max}}={r_max}$)",
        f"ToMe ($r={tome_r}$)",
    ]

    fig = plt.figure(figsize=(3.2 * n_cols, 3.2 * n_rows + 0.4))
    gs  = gridspec.GridSpec(
        n_rows, n_cols,
        hspace=0.04, wspace=0.04,
        top=0.94, bottom=0.02, left=0.02, right=0.98,
    )

    for row_i in range(n_rows):
        img_vis       = imgs_vis[row_i]
        ada_survived  = ada_survived_list[row_i]
        tome_survived = tome_survived_list[row_i]

        for col_i in range(n_cols):
            ax = fig.add_subplot(gs[row_i, col_i])
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)

            if col_i == 0:
                ax.imshow(img_vis)
                if row_i == 0:
                    ax.set_title(col_labels[0], fontsize=9,
                                 fontweight="bold", pad=4)

            elif col_i == 1:
                survived_overlay(ax, img_vis, ada_survived)
                if row_i == 0:
                    ax.set_title(col_labels[1], fontsize=9,
                                 fontweight="bold", pad=4)

            else:
                survived_overlay(ax, img_vis, tome_survived)
                if row_i == 0:
                    ax.set_title(col_labels[2], fontsize=9,
                                 fontweight="bold", pad=4)

    # 범례
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#44cc44", alpha=0.8, label="Survived"),
        Patch(facecolor="#ff4444", alpha=0.8, label="Merged"),
    ]
    fig.legend(handles=legend_elements, loc="lower center",
               ncol=2, fontsize=8, framealpha=0.9,
               bbox_to_anchor=(0.5, 0.0))

    fig.savefig(output_path, format="pdf", dpi=300, bbox_inches="tight")
    fig.savefig(output_path.replace(".pdf", ".png"),
                format="png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[완료] Figure 저장: {output_path}")


# ─────────────────────────────────────────
# Main
# ─────────────────────────────────────────

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[장치] {device}")

    print("[데이터] 이미지 로드 중...")
    positions = [int(p) for p in args.positions.split(",")] if args.positions else None
    imgs_vis, imgs_tensor = load_val_images(
        args.data_dir, args.pool, args.seed, positions
    )

    print(f"[모델] AdaMerge 로드 중 (r_max={args.r_max})...")
    model = ViTWithAdaMerge(
        r_max=args.r_max,
        alpha=args.alpha,
        stats_path=args.stats_path,
        pretrained=True,
    ).to(device)
    model.eval()

    ada_survived_list  = []
    tome_survived_list = []

    for i, (img_vis, img_tensor) in enumerate(zip(imgs_vis, imgs_tensor)):
        print(f"  이미지 {i+1}/{N_IMAGES} 처리 중...")
        ada_s  = get_adamerge_survived(model, img_tensor, device)
        tome_s = get_tome_survived(model, img_tensor, device, args.tome_r)
        ada_survived_list.append(ada_s)
        tome_survived_list.append(tome_s)
        print(f"    AdaMerge: {ada_s.sum()}/196, ToMe: {tome_s.sum()}/196")

    print("[Figure] 생성 중...")
    make_figure(
        imgs_vis, ada_survived_list, tome_survived_list,
        r_max=args.r_max, tome_r=args.tome_r,
        output_path=args.output,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",    type=str,   default="./data/imagenet1k")
    parser.add_argument("--stats_path",  type=str,   required=True)
    parser.add_argument("--r_max",       type=int,   default=16)
    parser.add_argument("--alpha",       type=float, default=3.0)
    parser.add_argument("--tome_r",      type=int,   default=5)
    parser.add_argument("--output",      type=str,   default="figure_merging_pattern.pdf")
    parser.add_argument("--seed",        type=int,   default=42)
    parser.add_argument("--pool",        type=int,   default=5,
                        help="seed로 샘플링할 전체 이미지 수")
    parser.add_argument("--positions",   type=str,   default=None,
                        help="사용할 위치 (쉼표 구분, 예: 0,1,4)")
    args = parser.parse_args()
    main(args)