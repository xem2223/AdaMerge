"""Evaluate (or calibrate) AdaMerge-on-official-ToMe.

Statistics MUST be recomputed for this pipeline (key-based similarity differs
from raw-feature similarity). Two-pass protocol, same as the paper:

  # pass 1 (default init mu=0.5 sigma=0.1)
  python evaluate_adamerge_official.py --collect_stats --subset 0.01 \
      --r_max 23 --pretrained --stats_out stats/official_rmax23_pass1.json

  # pass 2 (refine under pass-1 stats)
  python evaluate_adamerge_official.py --collect_stats --subset 0.01 \
      --r_max 23 --pretrained --stats stats/official_rmax23_pass1.json \
      --stats_out stats/official_rmax23_pass2.json

  # evaluation
  python evaluate_adamerge_official.py --r_max 23 --pretrained \
      --stats stats/official_rmax23_pass2.json \
      --output results/adamerge_official_rmax23.json
"""

import argparse
import json
import time
from pathlib import Path

import torch
import timm
from safetensors.torch import load_file
from torch.cuda.amp import autocast
from tqdm import tqdm

from adamerge_official import apply_adamerge_patch
from data_utils import build_imagenet_loader


def load_stats(path):
    with open(path) as f:
        raw = json.load(f)
    return {int(k): {"mu": float(v["mu"]), "sigma": float(v["sigma"])} for k, v in raw.items()}


def create_model(args, device):
    if args.checkpoint:
        model = timm.create_model("vit_base_patch16_224", pretrained=False)
        model.load_state_dict(load_file(args.checkpoint), strict=False)
    else:
        model = timm.create_model(args.model_name, pretrained=args.pretrained)
    stats = load_stats(args.stats) if args.stats else None
    apply_adamerge_patch(
        model, r_max=args.r_max, alpha=args.alpha, stats=stats,
        prop_attn=not args.no_prop_attn,
    )
    return model.to(device).eval()


@torch.no_grad()
def evaluate(model, loader, device):
    correct, total = 0, 0
    model.eval()
    for images, labels in tqdm(loader, desc="Evaluating AdaMerge-official"):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with autocast(enabled=device.type == "cuda"):
            logits = model(images)
        pred = logits.argmax(dim=-1)
        correct += (pred == labels).sum().item()
        total += labels.numel()
    return 100.0 * correct / total


@torch.no_grad()
def collect_stats(model, loader, device, num_layers=12):
    collector = {l: [] for l in range(num_layers)}
    model._adamerge_info["collect_stats"] = collector
    for images, _ in tqdm(loader, desc="Collecting layer statistics"):
        images = images.to(device, non_blocking=True)
        with autocast(enabled=device.type == "cuda"):
            _ = model(images)
    model._adamerge_info["collect_stats"] = None
    stats = {}
    for l, values in collector.items():
        t = torch.tensor(values, dtype=torch.float32)
        stats[l] = {"mu": t.mean().item(), "sigma": t.std(unbiased=True).item()}
    return stats


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
def measure_token_counts_and_flops(model, loader, device, n_batches=50):
    """FLOPs with correct attn/MLP accounting, on REAL images (adaptive r is
    input-dependent, so random noise would misestimate token counts)."""
    attn_counts, mlp_counts = [], []
    a_buf, m_buf = [], []
    attn_hooks = [
        blk.attn.register_forward_hook(lambda m, i, o: a_buf.append(o[0].shape[1]))
        for blk in model.blocks
    ]
    mlp_hooks = [
        blk.mlp.register_forward_hook(lambda m, i, o: m_buf.append(o.shape[1]))
        for blk in model.blocks
    ]
    r_logs = []
    for bi, (images, _) in enumerate(loader):
        if bi >= n_batches:
            break
        a_buf.clear(); m_buf.clear()
        images = images.to(device, non_blocking=True)
        _ = model(images)
        attn_counts.append(list(a_buf))
        mlp_counts.append(list(m_buf))
        r_logs.append(list(model._adamerge_info["r_log"]))
    for h in attn_hooks + mlp_hooks:
        h.remove()

    n = len(attn_counts)
    avg_attn = [sum(row[i] for row in attn_counts) / n for i in range(12)]
    avg_mlp = [sum(row[i] for row in mlp_counts) / n for i in range(12)]
    avg_r = [sum(row[i] for row in r_logs) / n for i in range(12)]

    dim = 768
    flops = 0.0
    for t_attn, t_mlp in zip(avg_attn, avg_mlp):
        flops += 2 * t_attn * t_attn * dim + 4 * t_attn * dim * dim
        flops += 2 * t_mlp * dim * (4 * dim)
    return [197.0] + avg_mlp, avg_r, flops / 1e9


def baseline_flops_vit_b16():
    t, dim = 197, 768
    return sum(2 * t * t * dim + 4 * t * dim * dim + 2 * t * dim * (4 * dim) for _ in range(12)) / 1e9


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model(args, device)

    if args.collect_stats:
        total_train = 1281167
        subset_size = int(total_train * args.subset)
        g = torch.Generator().manual_seed(args.seed)
        indices = torch.randperm(total_train, generator=g)[:subset_size].tolist()
        loader = build_imagenet_loader(args.data_dir, "train", args.batch_size,
                                       args.num_workers, indices=indices)
        stats = collect_stats(model, loader, device)
        Path(args.stats_out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.stats_out, "w") as f:
            json.dump({str(k): v for k, v in stats.items()}, f, indent=2)
        print(f"Saved stats to {args.stats_out}")
        print(json.dumps(stats, indent=2))
        return

    loader = build_imagenet_loader(args.data_dir, "validation", args.batch_size, args.num_workers)
    acc = evaluate(model, loader, device)
    thr = measure_throughput(model, device, args.batch_size, args.throughput_runs, args.warmup)
    flops_loader = build_imagenet_loader(args.data_dir, "validation", args.batch_size, args.num_workers)
    avg_counts, avg_r, flops = measure_token_counts_and_flops(model, flops_loader, device)
    base_flops = baseline_flops_vit_b16()

    result = {
        "setting": vars(args),
        "implementation": "AdaMerge on official-ToMe machinery (K-metric, size, prop-attn, attn-MLP position)",
        "adamerge_official": {
            "top1_acc": acc,
            "flops_g": flops,
            "flops_reduction_pct": 100.0 * (1.0 - flops / base_flops),
            "throughput_img_s": thr,
            "avg_block_output_counts": avg_counts,
            "avg_r_per_layer": avg_r,
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
    parser.add_argument("--r_max", type=int, default=23)
    parser.add_argument("--alpha", type=float, default=3.0)
    parser.add_argument("--stats", type=str, default=None)
    parser.add_argument("--no_prop_attn", action="store_true")
    parser.add_argument("--collect_stats", action="store_true")
    parser.add_argument("--stats_out", type=str, default="stats/official_stats.json")
    parser.add_argument("--subset", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--model_name", type=str, default="vit_base_patch16_224.augreg2_in21k_ft_in1k")
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--throughput_runs", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--output", type=str, default="results/adamerge_official.json")
    main(parser.parse_args())