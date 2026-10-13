"""
train.py
--------
AdaMerge ImageNet-1K 학습 파이프라인 (DDP + WandB + Warmup).

사용법:
  torchrun --nproc_per_node=3 train.py \
    --data_dir ./data \
    --stats    ./stats.json \
    --r_max    8 \
    --alpha    3.0 \
    --epochs   30 \
    --batch_size 256 \
    --lr       3e-5 \
    --warmup_epochs 3
"""

import os
import argparse
import json
import math
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from vit_adamerge import ViTWithAdaMerge
from salience import compute_token_salience


# ─────────────────────────────────────────
# DDP 유틸
# ─────────────────────────────────────────

def setup_ddp():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank

def cleanup_ddp():
    dist.destroy_process_group()

def is_main():
    return dist.get_rank() == 0


# ─────────────────────────────────────────
# WandB 초기화
# ─────────────────────────────────────────

def init_wandb(args):
    try:
        import wandb
        wandb.init(
            project="adamerge",
            name=f"rmax{args.r_max}_alpha{args.alpha}_lr{args.lr}",
            config=vars(args),
        )
        return wandb
    except ImportError:
        print("wandb 미설치. pip install wandb 후 재실행하세요.")
        return None


# ─────────────────────────────────────────
# 데이터 로더
# ─────────────────────────────────────────

class HFImageNetDataset(torch.utils.data.Dataset):
    def __init__(self, hf_dataset, transform=None):
        self.dataset = hf_dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item["image"].convert("RGB")
        label = item["label"]
        if self.transform:
            image = self.transform(image)
        return image, label


def build_loaders(data_dir, batch_size, num_workers, rank, world_size):
    from datasets import load_dataset

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.4, 0.4, 0.4),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5]),
    ])
    val_transform = transforms.Compose([
        transforms.Resize(249, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5]),
    ])

    if is_main():
        print("ImageNet-1k 로드 중...")

    hf_data = load_dataset("ILSVRC/imagenet-1k", cache_dir=data_dir)

    train_dataset = HFImageNetDataset(hf_data["train"],      transform=train_transform)
    val_dataset   = HFImageNetDataset(hf_data["validation"], transform=val_transform)

    train_sampler = DistributedSampler(
        train_dataset, num_replicas=world_size, rank=rank, shuffle=True
    )
    val_sampler = DistributedSampler(
        val_dataset, num_replicas=world_size, rank=rank, shuffle=False
    )

    per_gpu_batch = batch_size // world_size

    train_loader = DataLoader(
        train_dataset, batch_size=per_gpu_batch, sampler=train_sampler,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=per_gpu_batch, sampler=val_sampler,
        num_workers=num_workers, pin_memory=True,
    )

    if is_main():
        print(f"Train: {len(train_dataset):,}장 / Val: {len(val_dataset):,}장")
        print(f"GPU당 배치: {per_gpu_batch} × {world_size}GPU = 총 {batch_size}")

    return train_loader, val_loader, train_sampler


# ─────────────────────────────────────────
# Warmup + Cosine LR 스케줄러
# ─────────────────────────────────────────

def get_lr(epoch, args):
    """
    Warmup + Cosine Annealing LR 계산.

    warmup_epochs 동안 0 → lr로 선형 증가
    이후 cosine으로 감소
    """
    if epoch < args.warmup_epochs:
        return args.lr * (epoch + 1) / args.warmup_epochs
    else:
        progress = (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)
        return args.lr * 0.5 * (1 + math.cos(math.pi * progress))


def set_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


# ─────────────────────────────────────────
# 시각화용 정보 수집 (WandB 저장)
# ─────────────────────────────────────────

@torch.no_grad()
def collect_viz_info(model, imgs, device, num_layers=12):
    """
    실험 5 시각화에 필요한 정보 수집.

    수집 항목:
      - 레이어별 병합 토큰 수 (r_l)
      - 레이어별 평균 유사도
      - 레이어별 salience 분포
      - 이미지별 총 병합 수
    """
    raw_model = model.module if hasattr(model, "module") else model
    vit = raw_model.vit
    ada = raw_model.ada_merge

    from salience import compute_token_salience
    from adamerge import compute_salience_weighted_similarity, compute_adaptive_r

    import torch.nn.functional as F

    # patch embedding
    x = vit.patch_embed(imgs)
    cls = vit.cls_token.expand(x.shape[0], -1, -1)
    x = torch.cat([cls, x], dim=1)
    x = vit.pos_drop(x + vit.pos_embed)

    layer_r          = []   # 레이어별 r
    layer_mean_sim   = []   # 레이어별 평균 유사도
    layer_sal_mean   = []   # 레이어별 salience 평균
    layer_sal_std    = []   # 레이어별 salience std
    token_counts     = [x.shape[1]]  # 레이어별 토큰 수

    for l, block in enumerate(vit.blocks):
        # AdaMerge 적용 전 정보 수집
        patch = x[:, 1:, :]
        T = patch.shape[1]
        T_half = T // 2

        S = compute_token_salience(patch)
        x_A = patch[:, :T_half]
        x_B = patch[:, T_half:]
        sim = compute_salience_weighted_similarity(x_A, x_B, S)

        mu_l    = ada.stats[l]["mu"]
        sigma_l = ada.stats[l]["sigma"]
        r = compute_adaptive_r(sim, mu_l, sigma_l, ada.r_max, ada.alpha)

        mean_sim = sim.amax(dim=-1).mean().item()

        layer_r.append(r)
        layer_mean_sim.append(mean_sim)
        layer_sal_mean.append(S.mean().item())
        layer_sal_std.append(S.std().item())

        # AdaMerge + block 적용
        x = ada(x, layer_idx=l)
        x = block(x)
        token_counts.append(x.shape[1])

    total_merged = token_counts[0] - token_counts[-1]

    return {
        "layer_r":          layer_r,          # [12] 레이어별 병합 수
        "layer_mean_sim":   layer_mean_sim,   # [12] 레이어별 평균 유사도
        "layer_sal_mean":   layer_sal_mean,   # [12] 레이어별 salience 평균
        "layer_sal_std":    layer_sal_std,    # [12] 레이어별 salience std
        "token_counts":     token_counts,     # [13] 레이어별 잔존 토큰 수
        "total_merged":     total_merged,     # 총 병합 수
    }


# ─────────────────────────────────────────
# 평가
# ─────────────────────────────────────────

@torch.no_grad()
def evaluate(model, loader, device, world_size):
    model.eval()
    correct = torch.tensor(0, device=device)
    total   = torch.tensor(0, device=device)

    for imgs, labels in loader:
        imgs   = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with autocast():
            logits = model.module.forward(imgs)

        pred = logits.argmax(dim=-1)
        correct += (pred == labels).sum()
        total   += labels.size(0)

    dist.all_reduce(correct, op=dist.ReduceOp.SUM)
    dist.all_reduce(total,   op=dist.ReduceOp.SUM)

    return (correct / total * 100).item()


# ─────────────────────────────────────────
# 학습
# ─────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, criterion, scaler, device, epoch, world_size):
    model.train()
    total_loss = torch.tensor(0.0, device=device)
    correct    = torch.tensor(0,   device=device)
    total      = torch.tensor(0,   device=device)

    pbar = tqdm(loader, desc=f"Epoch {epoch:03d}", disable=not is_main())

    for imgs, labels in pbar:
        imgs   = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()

        with autocast():
            logits = model(imgs)
            loss   = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.detach()
        pred    = logits.argmax(dim=-1)
        correct += (pred == labels).sum()
        total   += labels.size(0)

        if is_main():
            pbar.set_postfix({
                "loss": f"{(total_loss / total).item():.4f}",
                "acc":  f"{(correct / total * 100).item():.2f}%"
            })

    dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
    dist.all_reduce(correct,    op=dist.ReduceOp.SUM)
    dist.all_reduce(total,      op=dist.ReduceOp.SUM)

    return (total_loss / total).item(), (correct / total * 100).item()


# ─────────────────────────────────────────
# 메인
# ─────────────────────────────────────────

def main(args):
    local_rank = setup_ddp()
    world_size = dist.get_world_size()
    device     = torch.device(f"cuda:{local_rank}")

    # WandB (rank 0만)
    wandb = None
    if is_main():
        print(f"DDP | GPU {world_size}장 | local_rank={local_rank}")
        os.makedirs(args.output, exist_ok=True)
        wandb = init_wandb(args)

    # 데이터
    train_loader, val_loader, train_sampler = build_loaders(
        args.data_dir, args.batch_size, args.num_workers, local_rank, world_size
    )

    # 모델
    model = ViTWithAdaMerge(
        r_max=args.r_max,
        alpha=args.alpha,
        stats_path=args.stats,
        pretrained=True,
    ).to(device)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    # ── Resume (이어서 학습) ──────────────────
    start_epoch = 1
    best_acc    = 0.0

    if args.resume and os.path.exists(args.resume):
        if is_main():
            print(f"Resume: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        model.module.load_state_dict(ckpt["model_state"])
        start_epoch = ckpt["epoch"] + 1
        best_acc    = ckpt.get("best_acc", ckpt.get("val_acc", 0.0))
        if is_main():
            print(f"  → epoch {ckpt['epoch']}부터 재시작 | best_acc={best_acc:.2f}%")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    scaler    = GradScaler()

    log = []
    if not args.resume:
        best_acc = 0.0

    if is_main():
        print(f"\n학습 시작 | r_max={args.r_max}, α={args.alpha}, "
              f"lr={args.lr}, warmup={args.warmup_epochs}epochs")
        print("=" * 65)

    for epoch in range(start_epoch, args.epochs + 1):
        train_sampler.set_epoch(epoch)

        # Warmup + Cosine LR 적용
        lr = get_lr(epoch - 1, args)
        set_lr(optimizer, lr)

        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion,
            scaler, device, epoch, world_size
        )
        val_acc = evaluate(model, val_loader, device, world_size)

        if is_main():
            print(f"Epoch {epoch:03d} | loss={train_loss:.4f} | "
                  f"train={train_acc:.2f}% | val={val_acc:.2f}% | lr={lr:.2e}")

            # ── 시각화 정보 수집 (5 epoch마다) ──
            viz_info = None
            if epoch % 5 == 0 or epoch == 1:
                sample_imgs, _ = next(iter(val_loader))
                sample_imgs = sample_imgs[:4].to(device)  # 4장만
                viz_info = collect_viz_info(model, sample_imgs, device)

            # ── WandB 로깅 ──────────────────────
            if wandb:
                log_dict = {
                    "epoch":      epoch,
                    "train_loss": train_loss,
                    "train_acc":  train_acc,
                    "val_acc":    val_acc,
                    "lr":         lr,
                }

                if viz_info:
                    # 레이어별 병합 수
                    for l, r in enumerate(viz_info["layer_r"]):
                        log_dict[f"layer_r/layer_{l:02d}"] = r

                    # 레이어별 유사도
                    for l, s in enumerate(viz_info["layer_mean_sim"]):
                        log_dict[f"layer_sim/layer_{l:02d}"] = s

                    # 레이어별 salience
                    for l, sm in enumerate(viz_info["layer_sal_mean"]):
                        log_dict[f"layer_sal_mean/layer_{l:02d}"] = sm

                    # 총 병합 수
                    log_dict["total_merged_tokens"] = viz_info["total_merged"]

                    # 토큰 수 변화 (레이어별)
                    for l, tc in enumerate(viz_info["token_counts"]):
                        log_dict[f"token_count/layer_{l:02d}"] = tc

                wandb.log(log_dict)

            # ── 로컬 로그 ───────────────────────
            entry = {
                "epoch": epoch, "train_loss": train_loss,
                "train_acc": train_acc, "val_acc": val_acc, "lr": lr,
            }
            if viz_info:
                entry["viz"] = viz_info
            log.append(entry)

            # ── Best 저장 ───────────────────────
            if val_acc > best_acc:
                best_acc  = val_acc
                ckpt_path = os.path.join(args.output, "best.pth")
                torch.save({
                    "epoch":           epoch,
                    "model_state":     model.module.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_lr":    lr,
                    "val_acc":         val_acc,
                    "args":            vars(args),
                }, ckpt_path)
                print(f"  → Best 저장: {val_acc:.2f}%")

            # ── 5 epoch마다 주기적 저장 (resume용) ──
            if epoch % 5 == 0:
                periodic_path = os.path.join(args.output, f"epoch_{epoch:03d}.pth")
                torch.save({
                    "epoch":           epoch,
                    "model_state":     model.module.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_lr":    lr,
                    "val_acc":         val_acc,
                    "best_acc":        best_acc,
                    "args":            vars(args),
                }, periodic_path)
                print(f"  → 체크포인트 저장: {periodic_path}")

    if is_main():
        log_path = os.path.join(args.output, "log.json")
        with open(log_path, "w") as f:
            json.dump(log, f, indent=2)
        print(f"\n학습 완료 | Best Val Acc: {best_acc:.2f}%")
        if wandb:
            wandb.finish()

    cleanup_ddp()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # 경로
    parser.add_argument("--data_dir",      type=str,   default="./data")
    parser.add_argument("--stats",         type=str,   default="./stats.json")
    parser.add_argument("--output",        type=str,   default="./checkpoints")

    # AdaMerge
    parser.add_argument("--r_max",         type=int,   default=8)
    parser.add_argument("--alpha",         type=float, default=3.0)

    # 학습
    parser.add_argument("--epochs",        type=int,   default=30)
    parser.add_argument("--warmup_epochs", type=int,   default=3)
    parser.add_argument("--batch_size",    type=int,   default=256)
    parser.add_argument("--lr",            type=float, default=3e-5)
    parser.add_argument("--weight_decay",  type=float, default=0.05)
    parser.add_argument("--num_workers",   type=int,   default=8)
    parser.add_argument("--resume",        type=str,   default=None,
                        help="이어서 학습할 체크포인트 경로")

    args = parser.parse_args()
    main(args)