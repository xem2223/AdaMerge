"""
measure_flops.py
----------------
AdaMerge vs Baseline FLOPs 및 Throughput 측정.
best.pth 체크포인트 로드 지원.

사용법:
  python measure_flops.py \
    --checkpoint ./checkpoints/best.pth \
    --stats      ./stats.json
"""

import torch
import timm
import time
import argparse
import json

from vit_adamerge import ViTWithAdaMerge


@torch.no_grad()
def measure_throughput(model, device, batch_size=64, n_runs=50, warmup=10):
    """Throughput 측정 (images/sec)"""
    model.eval()
    dummy = torch.randn(batch_size, 3, 224, 224).to(device)

    # Warmup
    for _ in range(warmup):
        _ = model(dummy)
    torch.cuda.synchronize()

    # 측정
    start = time.perf_counter()
    for _ in range(n_runs):
        _ = model(dummy)
    torch.cuda.synchronize()
    end = time.perf_counter()

    throughput = (batch_size * n_runs) / (end - start)
    return throughput


@torch.no_grad()
def measure_avg_tokens(model, device, n_samples=100):
    """
    실제 forward pass에서 레이어별 평균 토큰 수 측정.
    랜덤 입력 n_samples개에 대해 평균.
    """
    model.eval()
    all_token_counts = []

    for _ in range(n_samples):
        dummy = torch.randn(1, 3, 224, 224).to(device)

        # patch embedding
        x = model.vit.patch_embed(dummy)
        cls = model.vit.cls_token.expand(1, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = model.vit.pos_drop(x + model.vit.pos_embed)

        counts = [x.shape[1]]
        for l, block in enumerate(model.vit.blocks):
            x = model.ada_merge(x, layer_idx=l)
            x = block(x)
            counts.append(x.shape[1])

        all_token_counts.append(counts)

    # 레이어별 평균
    n_layers = len(all_token_counts[0])
    avg_counts = [
        sum(all_token_counts[i][l] for i in range(n_samples)) / n_samples
        for l in range(n_layers)
    ]
    return avg_counts


def estimate_flops(avg_token_counts, baseline_flops_g, d_model=768, n_heads=12):
    """
    레이어별 평균 토큰 수 기반 FLOPs 추정.

    각 Transformer block FLOPs:
      Attention:  2 * T^2 * D + 4 * T * D^2   (QKV proj + attention + out proj)
      MLP:        2 * T * D * 4D               (two linear layers, hidden=4D)
    """
    D = d_model
    flops = 0.0

    for l in range(len(avg_token_counts) - 1):
        T = avg_token_counts[l]   # AdaMerge 적용 후 토큰 수
        # Attention
        flops += 2 * T * T * D + 4 * T * D * D
        # MLP
        flops += 2 * T * D * (4 * D)

    return flops / 1e9  # GFLOPs


def load_checkpoint(checkpoint_path, stats_path, device):
    """best.pth에서 모델 로드"""
    print(f"체크포인트 로드: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device)

    saved_args = ckpt.get("args", {})
    r_max = saved_args.get("r_max", 8)
    alpha = saved_args.get("alpha", 3.0)

    print(f"  저장된 설정: r_max={r_max}, alpha={alpha}")
    print(f"  저장된 epoch: {ckpt['epoch']}")
    print(f"  저장된 val_acc: {ckpt['val_acc']:.2f}%")

    model = ViTWithAdaMerge(
        r_max=r_max,
        alpha=alpha,
        stats_path=stats_path,
        pretrained=False,
    ).to(device)

    # DDP로 저장된 경우 key에 "module." 접두사가 있을 수 있음
    state_dict = ckpt["model_state"]
    if any(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k[7:]: v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)
    model.eval()

    return model, r_max, alpha, ckpt['val_acc']


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # ── 1. Baseline ViT-B ─────────────────────
    print("=" * 55)
    print("[1/2] Baseline ViT-B/16")
    print("=" * 55)
    baseline = timm.create_model(
        "vit_base_patch16_224", pretrained=False
    ).to(device)
    baseline.eval()

    thr_base = measure_throughput(baseline, device, args.batch_size)

    # Baseline FLOPs 수식으로 계산
    T, D = 197, 768
    base_flops = 0.0
    for _ in range(12):
        base_flops += 2 * T * T * D + 4 * T * D * D  # Attention
        base_flops += 2 * T * D * (4 * D)              # MLP
    base_flops_g = base_flops / 1e9

    print(f"  FLOPs:      {base_flops_g:.2f} GFLOPs")
    print(f"  Throughput: {thr_base:.1f} img/s (batch={args.batch_size})")

    # ── 2. AdaMerge (best.pth) ─────────────────
    print(f"\n{'='*55}")
    print(f"[2/2] AdaMerge (best.pth)")
    print(f"{'='*55}")

    model, r_max, alpha, val_acc = load_checkpoint(
        args.checkpoint, args.stats, device
    )

    thr_ada = measure_throughput(model, device, args.batch_size)

    # 토큰 수 측정
    print(f"  토큰 수 측정 중 ({args.n_samples}샘플)...")
    avg_counts = measure_avg_tokens(model, device, args.n_samples)
    ada_flops_g = estimate_flops(avg_counts, base_flops_g)

    print(f"\n  FLOPs:             {ada_flops_g:.2f} GFLOPs")
    print(f"  FLOPs 감소율:      {(1 - ada_flops_g/base_flops_g)*100:.1f}%")
    print(f"  Throughput:        {thr_ada:.1f} img/s")
    print(f"  Throughput 향상:   {thr_ada/thr_base:.2f}×")
    print(f"  Val Acc (saved):   {val_acc:.2f}%")

    # ── 3. 레이어별 토큰 수 시각화 ────────────
    print(f"\n레이어별 평균 토큰 수:")
    print(f"  {'Layer':>6} {'Tokens':>7} {'비율':>7}  시각화")
    print(f"  {'-'*50}")
    for l, tc in enumerate(avg_counts):
        ratio = tc / avg_counts[0]
        bar = "█" * int(ratio * 25)
        label = "← AdaMerge 적용 전" if l == 0 else ""
        print(f"  {'입력' if l==0 else f'Layer {l-1:2d}':>6} "
              f"{tc:7.1f} {ratio:7.1%}  {bar} {label}")

    total_merged = avg_counts[0] - avg_counts[-1]
    print(f"\n  총 병합: {total_merged:.1f}개 "
          f"({total_merged/avg_counts[0]*100:.1f}% 감소)")

    # ── 4. 최종 요약 ───────────────────────────
    print(f"\n{'='*55}")
    print("최종 요약")
    print(f"{'='*55}")
    print(f"{'Method':<20} {'FLOPs':>10} {'Throughput':>12} {'속도':>8} {'Val Acc':>8}")
    print(f"{'-'*62}")
    print(f"{'Baseline':<20} {base_flops_g:>8.2f}G "
          f"{thr_base:>10.1f}/s {'1.00×':>8} {'81.8%':>8}")
    print(f"{'AdaMerge':<20} {ada_flops_g:>8.2f}G "
          f"{thr_ada:>10.1f}/s {thr_ada/thr_base:>7.2f}× {val_acc:>7.2f}%")

    # ── 5. 결과 저장 ───────────────────────────
    result = {
        "baseline": {
            "flops_g": base_flops_g,
            "throughput": thr_base,
        },
        "adamerge": {
            "checkpoint": args.checkpoint,
            "r_max": r_max,
            "alpha": alpha,
            "val_acc": val_acc,
            "flops_g": ada_flops_g,
            "flops_reduction_pct": (1 - ada_flops_g/base_flops_g)*100,
            "throughput": thr_ada,
            "throughput_speedup": thr_ada/thr_base,
            "avg_token_counts": avg_counts,
            "total_merged": total_merged,
        }
    }

    out_path = args.output
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n결과 저장: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str,
                        default="./checkpoints/best.pth",
                        help="best.pth 경로")
    parser.add_argument("--stats",      type=str,   default="./stats.json")
    parser.add_argument("--batch_size", type=int,   default=64)
    parser.add_argument("--n_samples",  type=int,   default=100,
                        help="토큰 수 측정에 사용할 샘플 수")
    parser.add_argument("--output",     type=str,   default="./flops_result.json")
    args = parser.parse_args()
    main(args)