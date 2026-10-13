"""
visualize_salience.py
---------------------
AdaMerge salience heatmap 시각화 스크립트.
HuggingFace Arrow 포맷 ImageNet-1k 지원.
"""

import os
import sys
import random
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image
from torchvision import transforms
from datasets import load_from_disk

from salience import compute_token_salience
from adamerge import AdaMerge, compute_salience_weighted_similarity, compute_adaptive_r, bipartite_merge
from vit_adamerge import ViTWithAdaMerge
from stats_utils import load_stats
import pyarrow as pa
import glob
import io

# ─────────────────────────────────────────
# 설정
# ─────────────────────────────────────────

SHOW_LAYERS = [0, 3, 7, 11]
N_IMAGES    = 5
PATCH_SIZE  = 16
IMG_SIZE    = 224
N_PATCHES   = IMG_SIZE // PATCH_SIZE  # 14

IMAGENET_MEAN = [0.5, 0.5, 0.5]
IMAGENET_STD  = [0.5, 0.5, 0.5]

CMAP = LinearSegmentedColormap.from_list(
    "salience", ["#ffffff", "#4488ff", "#ff2222"]
)


# ─────────────────────────────────────────
# 데이터 로드 (Arrow 포맷)
# ─────────────────────────────────────────
def get_val_images_from_arrow(imagenet_dir: str, n: int, seed: int = 42):
    """
    Arrow 파일에서 validation 이미지 직접 읽기.
    """
    arrow_dir = os.path.join(
        imagenet_dir,
        "ILSVRC___imagenet-1k",
        "default",
        "0.0.0",
        "49e2ee26f3810fb5a7536bbf732a7b07389a47b5",
    )

    val_files = sorted(glob.glob(os.path.join(arrow_dir, "imagenet-1k-validation-*.arrow")))
    if not val_files:
        raise FileNotFoundError(f"validation arrow 파일이 없습니다: {arrow_dir}")

    print(f"  validation arrow 파일 {len(val_files)}개 발견")

    # 전체 테이블 로드
    tables = []
    for f in val_files:
        with open(f, "rb") as source:
            reader = pa.ipc.open_stream(source)
            tables.append(reader.read_all())

    table = pa.concat_tables(tables)
    total = len(table)
    print(f"  총 validation 샘플 수: {total}")

    random.seed(seed)
    indices = random.sample(range(total), min(n, total))

    pil_images = []
    for idx in indices:
        row = table.slice(idx, 1)

        # image 컬럼 처리 (bytes 또는 dict 형태)
        img_col = row.column("image")[0]
        if hasattr(img_col, "as_py"):
            img_data = img_col.as_py()
        else:
            img_data = img_col

        if isinstance(img_data, dict):
            img_bytes = img_data.get("bytes") or img_data.get("path")
            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        elif isinstance(img_data, bytes):
            img = Image.open(io.BytesIO(img_data)).convert("RGB")
        else:
            raise ValueError(f"알 수 없는 image 타입: {type(img_data)}")

        label = row.column("label")[0].as_py()
        print(f"  idx={idx}, label={label}, size={img.size}")
        pil_images.append((img, idx))

    return pil_images


def preprocess_image(img_pil: Image.Image):
    """PIL 이미지 → (시각화용 PIL, 모델 입력 텐서)."""
    vis_transform = transforms.Compose([
        transforms.Resize(249, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
    ])
    tensor_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    img_vis    = vis_transform(img_pil)
    img_tensor = tensor_transform(img_vis).unsqueeze(0)  # [1, 3, 224, 224]

    return img_vis, img_tensor


# ─────────────────────────────────────────
# Salience 추출 Hook
# ─────────────────────────────────────────

class SalienceExtractor:
    def __init__(self, model: ViTWithAdaMerge):
        self.model    = model
        self.salience = {}
        self._hooks   = []

    def register(self):
        def hook(module, args, kwargs):
            x = args[0]
            layer_idx = kwargs.get("layer_idx", args[1] if len(args) > 1 else None)
            if layer_idx is None:
                return
            patch_tokens = x[:, 1:, :]
            with torch.no_grad():
                S = compute_token_salience(patch_tokens)
            self.salience[layer_idx] = S.detach().cpu()

        self._hooks.append(
            self.model.ada_merge.register_forward_pre_hook(hook, with_kwargs=True)
        )

    def remove(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def clear(self):
        self.salience.clear()


# ─────────────────────────────────────────
# 병합 결과 추출
# ─────────────────────────────────────────

def get_merged_token_map(model: ViTWithAdaMerge, img_tensor: torch.Tensor, device):
    """최종적으로 살아남은 패치 위치를 [196] bool 배열로 반환."""
    x = img_tensor.to(device)

    with torch.no_grad():
        x_emb = model.vit.patch_embed(x)
        cls_token = model.vit.cls_token.expand(x_emb.shape[0], -1, -1)
        x_emb = torch.cat([cls_token, x_emb], dim=1)
        x_emb = model.vit.pos_drop(x_emb + model.vit.pos_embed)

        indices = torch.arange(196, device=device).unsqueeze(0)  # [1, 196]

        for l, block in enumerate(model.vit.blocks):
            cls    = x_emb[:, :1, :]
            patches = x_emb[:, 1:, :]
            T_cur  = patches.shape[1]
            T_half = T_cur // 2

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
                keep_mask = ~merge_mask

                idx_A      = indices[:, :T_half]
                idx_B      = indices[:, T_half:]
                idx_A_keep = idx_A[keep_mask].unsqueeze(0)
                indices    = torch.cat([idx_A_keep, idx_B], dim=1)

                patches, S = bipartite_merge(patches, S, r, sim)

            x_emb = torch.cat([cls, patches], dim=1)
            x_emb = block(x_emb)

    survived_indices = indices[0].cpu().numpy()
    survived = np.zeros(196, dtype=bool)
    survived[survived_indices] = True
    return survived


# ─────────────────────────────────────────
# 시각화
# ─────────────────────────────────────────

def salience_to_heatmap(S: torch.Tensor, size: int = 224, n: int = 14) -> np.ndarray:
    S_np = S.squeeze().numpy()
    T    = S_np.shape[0]

    if T != n * n:
        S_tensor = torch.tensor(S_np).unsqueeze(0).unsqueeze(0)
        S_interp = F.interpolate(
            S_tensor, size=n * n, mode="linear", align_corners=False
        ).squeeze().numpy()
        hm = S_interp.reshape(n, n)
    else:
        hm = S_np.reshape(n, n)

    # percentile 기반 정규화 (상위 5% 기준으로 스케일)
    p_low  = np.percentile(hm, 20)
    p_high = np.percentile(hm, 95)
    hm = np.clip((hm - p_low) / (p_high - p_low + 1e-6), 0, 1)

    hm_pil = Image.fromarray((hm * 255).astype(np.uint8)).resize(
        (size, size), Image.BILINEAR
    )
    return np.array(hm_pil) / 255.0

def survived_to_heatmap(survived: np.ndarray, size: int = 224, n: int = 14) -> np.ndarray:
    """[196] bool → [size, size] float."""
    hm = survived.reshape(n, n).astype(np.float32)
    hm_pil = Image.fromarray((hm * 255).astype(np.uint8)).resize(
        (size, size), Image.NEAREST
    )
    return np.array(hm_pil) / 255.0

def add_grid(ax, n: int = 14, img_size: int = 224):
    patch_size = img_size / n  # 224/14 = 16px
    for i in range(n + 1):
        pos = i * patch_size - 0.5
        ax.axhline(pos, color="white", linewidth=0.4, alpha=0.6)
        ax.axvline(pos, color="white", linewidth=0.4, alpha=0.6)


def make_figure(images_vis, all_salience, all_survived, show_layers, output_path):
    n_rows = len(images_vis)
    n_cols = 1 + len(show_layers) + 1

    fig = plt.figure(figsize=(2.8 * n_cols, 3.2 * n_rows + 0.6))
    gs  = gridspec.GridSpec(
        n_rows, n_cols,
        hspace=0.02, wspace=0.05,
        top=0.93, bottom=0.04, left=0.02, right=0.98,
    )

    MERGE_CMAP = LinearSegmentedColormap.from_list(
        "merge", ["#ff4444", "#44cc44"]
    )

    for row_i, (img_vis, sal_dict, survived) in enumerate(
        zip(images_vis, all_salience, all_survived)
    ):
        for col_i in range(n_cols):
            ax = fig.add_subplot(gs[row_i, col_i])
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)

            if col_i == 0:
                ax.imshow(img_vis)
                if row_i == 0:
                    ax.set_title("Original", fontsize=9, fontweight="bold", pad=4)

            elif col_i <= len(show_layers):
                l = show_layers[col_i - 1]
                S = sal_dict.get(l)
                if S is not None:
                    hm = salience_to_heatmap(S)
                    ax.imshow(img_vis, alpha=0.45)
                    im = ax.imshow(hm, cmap=CMAP, alpha=0.70, vmin=0, vmax=1)
                    add_grid(ax)
                else:
                    ax.imshow(img_vis)
                if row_i == 0:
                    ax.set_title(f"Layer {l}", fontsize=9, fontweight="bold", pad=4)

            else:
                hm = survived_to_heatmap(survived)
                ax.imshow(img_vis, alpha=0.60)
                ax.imshow(hm, cmap=MERGE_CMAP, alpha=0.42, vmin=0, vmax=1)
                add_grid(ax)
                ax.text(
                    0.5, 0.02,
                    f"{survived.sum()}/196 tokens",
                    ha="center", va="bottom",
                    transform=ax.transAxes,
                    fontsize=7, color="white",
                    bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.5),
                )
                if row_i == 0:
                    ax.set_title("Survived Tokens", fontsize=9, fontweight="bold", pad=4)

    # Colorbar
    n_sal_cols = len(show_layers)
    col_width  = 1.0 / n_cols
    left       = col_width * 1      # 원본 열 하나 건너뜀
    width      = col_width * n_sal_cols  # salience 열 4개 범위
    cbar_ax = fig.add_axes([left, 0.01, width, 0.012])
    sm = plt.cm.ScalarMappable(cmap=CMAP, norm=plt.Normalize(0, 1))
    sm.set_array([])
    cb = fig.colorbar(sm, cax=cbar_ax, orientation="horizontal")
    cb.set_label("Salience Score", fontsize=15)
    cb.ax.tick_params(labelsize=6)

    fig.savefig(output_path, format="pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[완료] Figure 저장: {output_path}")


# ─────────────────────────────────────────
# 메인
# ─────────────────────────────────────────

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[장치] {device}")

    print("[모델] ViTWithAdaMerge 로드 중...")
    model = ViTWithAdaMerge(
        r_max=args.r_max,
        alpha=args.alpha,
        stats_path=args.stats_path,
        pretrained=True,
    ).to(device)
    model.eval()

    extractor = SalienceExtractor(model)
    extractor.register()

    print("[데이터] ImageNet val Arrow 로드 중...")
    val_items = get_val_images_from_arrow(
        args.imagenet_dir, N_IMAGES, seed=args.seed
    )

    images_vis   = []
    all_salience = []
    all_survived = []

    for img_pil, idx in val_items:
        img_vis, img_tensor = preprocess_image(img_pil)
        img_tensor = img_tensor.to(device)

        extractor.clear()
        with torch.no_grad():
            _ = model(img_tensor)

        sal_dict = {
            l: extractor.salience[l][0]
            for l in SHOW_LAYERS
            if l in extractor.salience
        }

        survived = get_merged_token_map(model, img_tensor, device)

        images_vis.append(img_vis)
        all_salience.append(sal_dict)
        all_survived.append(survived)
        print(f"  idx={idx}: {survived.sum()}/196 토큰 생존")

    extractor.remove()

    print("[Figure] 생성 중...")
    make_figure(
        images_vis=images_vis,
        all_salience=all_salience,
        all_survived=all_survived,
        show_layers=SHOW_LAYERS,
        output_path=args.output,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--imagenet_dir", type=str, default="./data/imagenet1k")
    parser.add_argument("--stats_path",   type=str, default="./stats.json")
    parser.add_argument("--r_max",        type=int, default=16)
    parser.add_argument("--alpha",        type=float, default=3.0)
    parser.add_argument("--output",       type=str, default="figure_salience.pdf")
    parser.add_argument("--seed",         type=int, default=42)
    args = parser.parse_args()
    main(args)