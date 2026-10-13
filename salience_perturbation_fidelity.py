import os
import io
import glob
import json
import argparse
import random
from collections import defaultdict

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image

import timm
import pyarrow as pa
from safetensors.torch import load_file
from tqdm import tqdm
import numpy as np

from salience import compute_token_salience


# ─────────────────────────────────────────
# Data loading: local Arrow ImageNet-1k
# ─────────────────────────────────────────

TRANSFORM = transforms.Compose([
    transforms.Resize(249, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3),
])


class ArrowDataset(torch.utils.data.Dataset):
    def __init__(self, table, indices=None, transform=None):
        self.table = table
        self.indices = list(range(len(table))) if indices is None else indices
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        row = self.table.slice(int(self.indices[i]), 1)
        img_data = row.column("image")[0].as_py()
        label = row.column("label")[0].as_py()

        img = Image.open(io.BytesIO(
            img_data["bytes"] if isinstance(img_data, dict) else img_data
        )).convert("RGB")

        if self.transform:
            img = self.transform(img)

        return img, label


def load_arrow(data_dir, split="validation"):
    arrow_dir = os.path.join(
        data_dir,
        "ILSVRC___imagenet-1k",
        "default",
        "0.0.0",
        "49e2ee26f3810fb5a7536bbf732a7b07389a47b5",
    )

    files = sorted(
        glob.glob(os.path.join(arrow_dir, f"imagenet-1k-{split}-*.arrow"))
    )

    if not files:
        raise FileNotFoundError(f"{split} arrow 없음: {arrow_dir}")

    tables = []
    for f in files:
        with open(f, "rb") as src:
            tables.append(pa.ipc.open_stream(src).read_all())

    return pa.concat_tables(tables)


# ─────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────

def load_model(device):
    """vit_adamerge.py와 동일한 safetensors 로드 방식"""
    model = timm.create_model("vit_base_patch16_224", pretrained=False)

    cache_path = os.path.expanduser(
        "~/.cache/huggingface/hub/"
        "models--timm--vit_base_patch16_224.augreg2_in21k_ft_in1k/"
        "snapshots/063c6c38a5d8510b2e57df480445e94b231dad2c/model.safetensors"
    )

    state_dict = load_file(cache_path)
    model.load_state_dict(state_dict, strict=False)
    return model.to(device).eval()


# ─────────────────────────────────────────
# Forward utilities
# ─────────────────────────────────────────

@torch.no_grad()
def forward_full(model, imgs):
    """Standard ViT forward."""
    return model(imgs)


@torch.no_grad()
def forward_until_layer(model, imgs, layer_idx):
    """
    Forward ViT until the input of transformer block `layer_idx`.

    Returns:
        x: token sequence entering block layer_idx, shape [B, 197, D]
    """
    x = model.patch_embed(imgs)
    cls_token = model.cls_token.expand(x.shape[0], -1, -1)
    x = torch.cat([cls_token, x], dim=1)
    x = model.pos_drop(x + model.pos_embed)

    for l in range(layer_idx):
        x = model.blocks[l](x)

    return x


@torch.no_grad()
def forward_from_layer(model, x, layer_idx):
    """
    Continue ViT forward from block `layer_idx`.

    Args:
        x: token sequence entering block layer_idx, shape [B, 197, D]
    """
    for l in range(layer_idx, len(model.blocks)):
        x = model.blocks[l](x)

    x = model.norm(x)
    x = x[:, 0]
    x = model.head(x)
    return x


# ─────────────────────────────────────────
# Perturbation
# ─────────────────────────────────────────

def perturb_tokens_by_salience(
    x,
    salience,
    frac=0.10,
    mode="top",
    perturb_type="mean",
):
    """
    Perturb top- or bottom-salience patch tokens.

    Args:
        x: token sequence with CLS, shape [B, 1+N, D]
        salience: patch-token salience, shape [B, N]
        frac: fraction of patch tokens to perturb
        mode: "top" or "bottom"
        perturb_type:
            - "mean": replace selected tokens with per-image mean patch feature
            - "zero": replace selected tokens with zeros
            - "shuffle": replace selected tokens with randomly permuted patch tokens

    Returns:
        x_perturbed
    """
    assert mode in ["top", "bottom"]
    assert perturb_type in ["mean", "zero", "shuffle"]

    B, T, D = x.shape
    N = T - 1
    k = max(1, int(round(N * frac)))

    patch = x[:, 1:, :]  # [B, N, D]

    if mode == "top":
        idx = salience.topk(k, dim=1, largest=True).indices
    else:
        idx = salience.topk(k, dim=1, largest=False).indices

    x_out = x.clone()
    patch_out = x_out[:, 1:, :]

    if perturb_type == "mean":
        replacement = patch.mean(dim=1, keepdim=True).expand(-1, k, -1)

    elif perturb_type == "zero":
        replacement = torch.zeros(B, k, D, device=x.device, dtype=x.dtype)

    elif perturb_type == "shuffle":
        replacement_list = []
        for b in range(B):
            perm = torch.randperm(N, device=x.device)[:k]
            replacement_list.append(patch[b, perm, :])
        replacement = torch.stack(replacement_list, dim=0)

    patch_out.scatter_(
        dim=1,
        index=idx.unsqueeze(-1).expand(-1, -1, D),
        src=replacement,
    )

    return x_out


# ─────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────

@torch.no_grad()
def evaluate_salience_perturbation(
    model,
    loader,
    device,
    layers,
    frac=0.10,
    perturb_type="mean",
):
    """
    For each layer:
      1. Run baseline model and store baseline predicted class.
      2. At the chosen layer, compute salience.
      3. Perturb top-salience tokens or bottom-salience tokens.
      4. Continue forward and measure drop in baseline-class confidence.

    Metrics:
      - confidence_drop: p_base(y_base) - p_perturbed(y_base)
      - flip_rate: whether perturbed prediction differs from baseline prediction
      - gt_acc_after: ground-truth accuracy after perturbation
    """
    stats = {
        l: {
            "top_drop": [],
            "bottom_drop": [],
            "top_flip": [],
            "bottom_flip": [],
            "top_gt_correct": [],
            "bottom_gt_correct": [],
        }
        for l in layers
    }

    baseline_correct = []
    baseline_conf_all = []

    for imgs, labels in tqdm(loader, desc="Salience perturbation fidelity"):
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # Baseline prediction
        logits_base = forward_full(model, imgs)
        prob_base = F.softmax(logits_base, dim=-1)
        pred_base = prob_base.argmax(dim=-1)
        conf_base = prob_base.gather(1, pred_base[:, None]).squeeze(1)

        baseline_correct.extend((pred_base == labels).float().cpu().tolist())
        baseline_conf_all.extend(conf_base.cpu().tolist())

        for layer_idx in layers:
            # Forward to target layer
            x_layer = forward_until_layer(model, imgs, layer_idx)

            # Compute salience on patch tokens entering this block
            patch_tokens = x_layer[:, 1:, :]
            salience = compute_token_salience(patch_tokens)  # [B, N]

            # Bottom-salience perturbation
            x_bottom = perturb_tokens_by_salience(
                x_layer,
                salience,
                frac=frac,
                mode="bottom",
                perturb_type=perturb_type,
            )
            logits_bottom = forward_from_layer(model, x_bottom, layer_idx)
            prob_bottom = F.softmax(logits_bottom, dim=-1)
            pred_bottom = prob_bottom.argmax(dim=-1)
            conf_bottom_on_base = prob_bottom.gather(
                1, pred_base[:, None]
            ).squeeze(1)

            bottom_drop = conf_base - conf_bottom_on_base
            bottom_flip = (pred_bottom != pred_base).float()
            bottom_gt_correct = (pred_bottom == labels).float()

            # Top-salience perturbation
            x_top = perturb_tokens_by_salience(
                x_layer,
                salience,
                frac=frac,
                mode="top",
                perturb_type=perturb_type,
            )
            logits_top = forward_from_layer(model, x_top, layer_idx)
            prob_top = F.softmax(logits_top, dim=-1)
            pred_top = prob_top.argmax(dim=-1)
            conf_top_on_base = prob_top.gather(
                1, pred_base[:, None]
            ).squeeze(1)

            top_drop = conf_base - conf_top_on_base
            top_flip = (pred_top != pred_base).float()
            top_gt_correct = (pred_top == labels).float()

            stats[layer_idx]["bottom_drop"].extend(bottom_drop.cpu().tolist())
            stats[layer_idx]["bottom_flip"].extend(bottom_flip.cpu().tolist())
            stats[layer_idx]["bottom_gt_correct"].extend(bottom_gt_correct.cpu().tolist())

            stats[layer_idx]["top_drop"].extend(top_drop.cpu().tolist())
            stats[layer_idx]["top_flip"].extend(top_flip.cpu().tolist())
            stats[layer_idx]["top_gt_correct"].extend(top_gt_correct.cpu().tolist())

    summary = {
        "setting": {
            "perturb_frac": frac,
            "perturb_type": perturb_type,
            "layers": layers,
        },
        "baseline": {
            "top1_acc": float(np.mean(baseline_correct) * 100),
            "mean_confidence": float(np.mean(baseline_conf_all)),
            "n": int(len(baseline_correct)),
        },
        "layers": {},
    }

    for l in layers:
        layer_stats = stats[l]

        top_drop = np.array(layer_stats["top_drop"], dtype=np.float64)
        bottom_drop = np.array(layer_stats["bottom_drop"], dtype=np.float64)

        top_flip = np.array(layer_stats["top_flip"], dtype=np.float64)
        bottom_flip = np.array(layer_stats["bottom_flip"], dtype=np.float64)

        top_acc = np.array(layer_stats["top_gt_correct"], dtype=np.float64)
        bottom_acc = np.array(layer_stats["bottom_gt_correct"], dtype=np.float64)

        summary["layers"][str(l)] = {
            "top_salience": {
                "confidence_drop_mean": float(top_drop.mean()),
                "confidence_drop_std": float(top_drop.std()),
                "flip_rate_pct": float(top_flip.mean() * 100),
                "gt_top1_acc_pct": float(top_acc.mean() * 100),
            },
            "bottom_salience": {
                "confidence_drop_mean": float(bottom_drop.mean()),
                "confidence_drop_std": float(bottom_drop.std()),
                "flip_rate_pct": float(bottom_flip.mean() * 100),
                "gt_top1_acc_pct": float(bottom_acc.mean() * 100),
            },
            "top_minus_bottom_drop": float(top_drop.mean() - bottom_drop.mean()),
            "drop_ratio_top_over_bottom": float(
                top_drop.mean() / max(bottom_drop.mean(), 1e-8)
            ),
            "n": int(len(top_drop)),
        }

    return summary


# ─────────────────────────────────────────
# Main
# ─────────────────────────────────────────

def print_summary(summary):
    print("\n=== Salience Perturbation Fidelity ===")
    print(f"Perturb frac: {summary['setting']['perturb_frac']}")
    print(f"Perturb type: {summary['setting']['perturb_type']}")
    print(
        f"Baseline Top-1: {summary['baseline']['top1_acc']:.2f}% | "
        f"Mean conf: {summary['baseline']['mean_confidence']:.4f} | "
        f"n={summary['baseline']['n']}"
    )

    print("\nLayer | Bottom drop | Top drop | Top-Bottom | Bottom flip | Top flip")
    print("---------------------------------------------------------------------")

    for l_str, res in summary["layers"].items():
        bottom = res["bottom_salience"]
        top = res["top_salience"]

        print(
            f"{int(l_str):>5} | "
            f"{bottom['confidence_drop_mean']:.4f} ± {bottom['confidence_drop_std']:.4f} | "
            f"{top['confidence_drop_mean']:.4f} ± {top['confidence_drop_std']:.4f} | "
            f"{res['top_minus_bottom_drop']:.4f} | "
            f"{bottom['flip_rate_pct']:.2f}% | "
            f"{top['flip_rate_pct']:.2f}%"
        )


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print("Loading ImageNet-1k Arrow validation set...")
    table = load_arrow(args.data_dir, split="validation")

    n_total = len(table)
    indices = list(range(n_total))
    random.shuffle(indices)
    indices = indices[:args.n_samples]

    dataset = ArrowDataset(table, indices=indices, transform=TRANSFORM)

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    print(f"Validation samples: {len(dataset)}")
    print(f"Layers: {args.layers}")
    print(f"Perturb fraction: {args.perturb_frac}")
    print(f"Perturb type: {args.perturb_type}")

    print("Loading ViT-B/16 model...")
    model = load_model(device)

    summary = evaluate_salience_perturbation(
        model=model,
        loader=loader,
        device=device,
        layers=args.layers,
        frac=args.perturb_frac,
        perturb_type=args.perturb_type,
    )

    print_summary(summary)

    with open(args.output, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved to: {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--n_samples", type=int, default=1000)
    parser.add_argument("--layers", type=int, nargs="+", default=[0, 3, 7, 11])
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--perturb_frac",
        type=float,
        default=0.10,
        help="Fraction of patch tokens to perturb, e.g., 0.10 means 10%.",
    )
    parser.add_argument(
        "--perturb_type",
        type=str,
        default="mean",
        choices=["mean", "zero", "shuffle"],
        help="How to perturb selected tokens.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="salience_perturbation_fidelity.json",
    )

    args = parser.parse_args()
    main(args)