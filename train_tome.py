"""
train_tome.py
-------------
ToMe fine-tuning 파이프라인 (DDP).
외부 tome 패키지 없이 핵심 로직 직접 구현.
AdaMerge와 동일한 실험 환경으로 공정한 비교.

사용법:
  torchrun --nproc_per_node=3 train_tome.py \
    --data_dir ./data \
    --r        8 \
    --lr       5e-6 \
    --warmup_epochs 3 \
    --epochs   30 \
    --batch_size 256
"""

import os
import argparse
import json
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import timm


# ─────────────────────────────────────────
# ToMe 핵심 로직 (직접 구현)
# ─────────────────────────────────────────

def bipartite_soft_matching(
    metric: torch.Tensor,
    r: int,
) -> torch.Tensor:
    """
    ToMe의 Bipartite Soft Matching.
    단순 코사인 유사도로 병합 쌍 선택.

    Args:
        metric: [N, T, D] 토큰 feature
        r:      병합할 토큰 수

    Returns:
        merge: [N, T-r, D] 병합 후 토큰
    """
    N, T, D = metric.shape
    T_half = T // 2

    x_A = metric[:, :T_half]   # [N, T/2, D]
    x_B = metric[:, T_half:]   # [N, T/2, D]

    if r <= 0:
        return metric

    # 코사인 유사도 (salience 가중 없음 - ToMe 방식)
    a_norm = F.normalize(x_A, dim=-1)
    b_norm = F.normalize(x_B, dim=-1)
    scores = torch.bmm(a_norm, b_norm.transpose(1, 2))  # [N, T/2, T/2]

    # 각 A 토큰의 가장 유사한 B 인덱스
    best_match = scores.argmax(dim=-1)         # [N, T/2]

    # 상위 r개 A 토큰 선택
    top_scores = scores.amax(dim=-1)           # [N, T/2]
    _, top_idx = top_scores.topk(r, dim=-1)   # [N, r]

    # 병합할 A 토큰 마스크
    merge_mask = torch.zeros(N, T_half, dtype=torch.bool, device=metric.device)
    merge_mask.scatter_(1, top_idx, True)

    # 병합할 A의 B 대상 인덱스
    b_idx = best_match.gather(1, top_idx)      # [N, r]

    # 단순 평균 병합 (ToMe 방식)
    x_A_merge = x_A.gather(
        1, top_idx.unsqueeze(-1).expand(-1, -1, D)
    )  # [N, r, D]
    x_B_target = x_B.gather(
        1, b_idx.unsqueeze(-1).expand(-1, -1, D)
    )  # [N, r, D]

    # ToMe: 단순 평균
    x_merged = (x_A_merge + x_B_target) / 2   # [N, r, D]

    # B에 병합 결과 scatter
    x_B_out = x_B.clone()
    x_B_out.scatter_(
        1,
        b_idx.unsqueeze(-1).expand(-1, -1, D),
        x_merged
    )

    # 병합 안 된 A 토큰 보존
    keep_mask = ~merge_mask
    x_A_keep = x_A[keep_mask].view(N, T_half - r, D)

    return torch.cat([x_A_keep, x_B_out], dim=1)  # [N, T-r, D]


class ToMeBlock(nn.Module):
    """
    ToMe가 적용된 ViT Block.
    기존 block에 bipartite_soft_matching을 삽입.
    """
    def __init__(self, block: nn.Module, r: int):
        super().__init__()
        self.block = block
        self.r = r

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # CLS 토큰 분리
        cls_token = x[:, :1, :]
        patch_tokens = x[:, 1:, :]

        # ToMe 병합 (patch token에만 적용)
        if self.r > 0 and patch_tokens.shape[1] > self.r * 2:
            patch_tokens = bipartite_soft_matching(patch_tokens, self.r)

        # CLS 재결합 후 Transformer block
        x = torch.cat([cls_token, patch_tokens], dim=1)
        x = self.block(x)
        return x


def build_tome_model(r: int, pretrained: bool = True):
    """
    timm ViT-B에 ToMe 직접 적용.
    AdaMerge와 동일한 pretrained 모델 사용.
    """
    model = timm.create_model("vit_base_patch16_224", pretrained=False)

    if pretrained:
        import os
        from safetensors.torch import load_file
        cache_path = os.path.expanduser(
            "~/.cache/huggingface/hub/"
            "models--timm--vit_base_patch16_224.augreg2_in21k_ft_in1k/"
            "snapshots/063c6c38a5d8510b2e57df480445e94b231dad2c/model.safetensors"
        )
        state_dict = load_file(cache_path)
        model.load_state_dict(state_dict, strict=False)
        print(f"  가중치 로드 완료")

    # 각 block에 ToMe 삽입
    new_blocks = nn.Sequential(*[
        ToMeBlock(block, r) for block in model.blocks
    ])
    model.blocks = new_blocks

    return model


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
# WandB
# ─────────────────────────────────────────

def init_wandb(args):
    try:
        import wandb
        wandb.init(
            project="adamerge",
            name=f"tome_r{args.tome_r}_lr{args.lr}",
            config=vars(args),
        )
        return wandb
    except ImportError:
        print("wandb 미설치. 로컬 로그만 저장.")
        return None


# ─────────────────────────────────────────
# 데이터 로더 (AdaMerge train.py와 동일)
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
# Warmup + Cosine LR (AdaMerge와 동일)
# ─────────────────────────────────────────

def get_lr(epoch, args):
    if epoch < args.warmup_epochs:
        return args.lr * (epoch + 1) / args.warmup_epochs
    progress = (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)
    return args.lr * 0.5 * (1 + math.cos(math.pi * progress))

def set_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


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
            logits = model.module(imgs)

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
                "loss": f"{(total_loss/total).item():.4f}",
                "acc":  f"{(correct/total*100).item():.2f}%"
            })

    dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
    dist.all_reduce(correct,    op=dist.ReduceOp.SUM)
    dist.all_reduce(total,      op=dist.ReduceOp.SUM)

    return (total_loss/total).item(), (correct/total*100).item()


# ─────────────────────────────────────────
# 메인
# ─────────────────────────────────────────

def main(args):
    local_rank = setup_ddp()
    world_size = dist.get_world_size()
    device     = torch.device(f"cuda:{local_rank}")

    wandb = None
    if is_main():
        print(f"DDP | GPU {world_size}장 | ToMe r={args.tome_r}")
        os.makedirs(args.output, exist_ok=True)
        wandb = init_wandb(args)

    train_loader, val_loader, train_sampler = build_loaders(
        args.data_dir, args.batch_size, args.num_workers, local_rank, world_size
    )

    # 모델 (AdaMerge와 동일한 pretrained)
    model = build_tome_model(r=args.tome_r, pretrained=True).to(device)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    scaler    = GradScaler()

    # Resume
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
            print(f"  → epoch {ckpt['epoch']}부터 재시작")

    log = []

    if is_main():
        print(f"\n학습 시작 | ToMe r={args.tome_r}, lr={args.lr}, "
              f"warmup={args.warmup_epochs}epochs")
        print("=" * 60)

    for epoch in range(start_epoch, args.epochs + 1):
        train_sampler.set_epoch(epoch)

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

            if wandb:
                wandb.log({
                    "epoch": epoch, "train_loss": train_loss,
                    "train_acc": train_acc, "val_acc": val_acc, "lr": lr,
                })

            log.append({
                "epoch": epoch, "train_loss": train_loss,
                "train_acc": train_acc, "val_acc": val_acc, "lr": lr,
            })

            # Best 저장
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save({
                    "epoch": epoch,
                    "model_state": model.module.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_lr": lr,
                    "val_acc": val_acc,
                    "best_acc": best_acc,
                    "args": vars(args),
                }, os.path.join(args.output, "tome_best.pth"))
                print(f"  → Best 저장: {val_acc:.2f}%")

            # 5 epoch마다 저장
            if epoch % 5 == 0:
                torch.save({
                    "epoch": epoch,
                    "model_state": model.module.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_lr": lr,
                    "val_acc": val_acc,
                    "best_acc": best_acc,
                    "args": vars(args),
                }, os.path.join(args.output, f"tome_epoch_{epoch:03d}.pth"))
                print(f"  → 체크포인트 저장: epoch {epoch}")

    if is_main():
        log_path = os.path.join(args.output, "tome_log.json")
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
    parser.add_argument("--output",        type=str,   default="./checkpoints")
    parser.add_argument("--resume",        type=str,   default=None)

    # ToMe
    parser.add_argument("--tome_r",        type=int,   default=8)

    # 학습 (AdaMerge와 동일)
    parser.add_argument("--epochs",        type=int,   default=30)
    parser.add_argument("--warmup_epochs", type=int,   default=3)
    parser.add_argument("--batch_size",    type=int,   default=256)
    parser.add_argument("--lr",            type=float, default=5e-6)
    parser.add_argument("--weight_decay",  type=float, default=0.05)
    parser.add_argument("--num_workers",   type=int,   default=8)

    args = parser.parse_args()
    main(args)