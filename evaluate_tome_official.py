"""Official-ToMe evaluation on ImageNet-1k validation.

Uses the official facebookresearch/ToMe implementation (bipartite soft matching
on attention keys, weighted-average merging with token-size tracking, and
proportional attention), patched onto the same timm ViT-B/16 checkpoint,
preprocessing, and measurement protocol as the AdaMerge experiments.

Setup (once):
    git clone https://github.com/facebookresearch/ToMe.git
    pip install -e ./ToMe

Run:
    python evaluate_tome_official.py \
        --data_dir ./data --r 8 --pretrained \
        --output results/tome_official_r8.json
"""

import argparse
import json
import time
from pathlib import Path

import torch
import timm
import tome
from safetensors.torch import load_file
from torch.cuda.amp import autocast
from tqdm import tqdm

from data_utils import build_imagenet_loader


def create_vit_b16(model_name, pretrained, checkpoint, device):
    if checkpoint:
        model = timm.create_model("vit_base_patch16_224", pretrained=False)
        model.load_state_dict(load_file(checkpoint), strict=False)
    else:
        model = timm.create_model(model_name, pretrained=pretrained)
    return model.to(device).eval()


def build_official_tome_model(args, device):
    model = create_vit_b16(args.model_name, args.pretrained, args.checkpoint, device)
    # Official patch: merging between attention and MLP, similarity on attention
    # keys (metric = k.mean(heads)), weighted-average merge with size tracking,
    # proportional attention enabled (recommended for off-the-shelf evaluation).
    tome.patch.timm(model, prop_attn=not args.no_prop_attn)
    model.r = args.r
    return model.eval()


@torch.no_grad()
def evaluate(model, loader, device):
    correct, total = 0, 0
    model.eval()
    for images, labels in tqdm(loader, desc=f"Evaluating official ToMe r={model.r}"):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with autocast(enabled=device.type == "cuda"):
            logits = model(images)
        pred = logits.argmax(dim=-1)
        correct += (pred == labels).sum().item()
        total += labels.numel()
    return 100.0 * correct / total


@torch.no_grad()
def measure_throughput(model, device, batch_size, n_runs, warmup):
    dummy = torch.randn(batch_size, 3, 224, 224, device=device)
    for _ in range(warmup):
        _ = model(dummy)
    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(n_runs):
        _ = model(dummy)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return batch_size * n_runs / (time.perf_counter() - start)


@torch.no_grad()
def measure_token_counts_and_flops(model, device):
    """Estimate FLOPs with attention/MLP token counts accounted separately.

    Official ToMe merges *between* attention and the MLP inside each block, so
    within block l attention runs on n_in tokens while the MLP runs on
    n_in - r tokens. Token counts are deterministic for fixed r, so a single
    forward pass suffices.
    """
    attn_counts, mlp_counts = [], []
    attn_hooks = [
        blk.attn.register_forward_hook(lambda m, i, o: attn_counts.append(o[0].shape[1]))
        for blk in model.blocks
    ]
    mlp_hooks = [
        blk.mlp.register_forward_hook(lambda m, i, o: mlp_counts.append(o.shape[1]))
        for blk in model.blocks
    ]
    _ = model(torch.randn(1, 3, 224, 224, device=device))
    for h in attn_hooks + mlp_hooks:
        h.remove()

    dim = 768
    flops = 0.0
    for t_attn, t_mlp in zip(attn_counts, mlp_counts):
        flops += 2 * t_attn * t_attn * dim + 4 * t_attn * dim * dim  # attention
        flops += 2 * t_mlp * dim * (4 * dim)                          # MLP
    block_out_counts = [197] + mlp_counts
    return block_out_counts, flops / 1e9


def baseline_flops_vit_b16():
    t, dim = 197, 768
    return sum(
        2 * t * t * dim + 4 * t * dim * dim + 2 * t * dim * (4 * dim)
        for _ in range(12)
    ) / 1e9


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader = build_imagenet_loader(args.data_dir, "validation", args.batch_size, args.num_workers)
    model = build_official_tome_model(args, device)

    acc = evaluate(model, loader, device)
    throughput = measure_throughput(model, device, args.batch_size, args.throughput_runs, args.warmup)
    avg_counts, flops = measure_token_counts_and_flops(model, device)
    base_flops = baseline_flops_vit_b16()

    result = {
        "setting": vars(args),
        "implementation": "official facebookresearch/ToMe (key metric, wavg merge, prop attn)",
        "tome_official": {
            "top1_acc": acc,
            "flops_g": flops,
            "flops_reduction_pct": 100.0 * (1.0 - flops / base_flops),
            "throughput_img_s": throughput,
            "block_output_token_counts": avg_counts,
            "total_merged_tokens": avg_counts[0] - avg_counts[-1],
        },
    }
    print(json.dumps(result, indent=2))
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--r", type=int, default=8)
    parser.add_argument("--no_prop_attn", action="store_true",
                        help="Disable proportional attention (official default is ON).")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--model_name", type=str, default="vit_base_patch16_224.augreg2_in21k_ft_in1k")
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--throughput_runs", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--output", type=str, default="results/tome_official.json")
    main(parser.parse_args())