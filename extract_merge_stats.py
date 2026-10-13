"""
extract_merge_stats.py  (세미 세팅 맞춤)
─────────────────────────────────────────
ImageNet-1k validation set을 forward pass하면서
각 r 값에 대해 두 통계를 수집합니다:
  (1) 병합 pair의 평균 cosine similarity  ⟨cos⟩_r
  (2) 병합 pair의 평균 salience asymmetry ⟨(s_i - s_j)²⟩_r

실행 방법:
  python extract_merge_stats.py \
      --data_dir ./data/imagenet1k \
      --batch_size 64 \
      --num_batches 20 \
      --r_values 3 4 5 6 7 8
"""

import os
import glob
import io
import argparse
import json
import numpy as np
from collections import defaultdict

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import pyarrow as pa
import timm
from safetensors.torch import load_file

# ── Arrow 데이터 로더 (세미 세팅) ──────────────────────────────────────────────

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


def load_arrow(data_dir: str, split: str = "validation"):
    arrow_dir = os.path.join(
        data_dir,
        "ILSVRC___imagenet-1k", "default", "0.0.0",
        "49e2ee26f3810fb5a7536bbf732a7b07389a47b5",
    )
    files = sorted(glob.glob(
        os.path.join(arrow_dir, f"imagenet-1k-{split}-*.arrow")
    ))
    if not files:
        raise FileNotFoundError(f"{split} arrow 파일 없음: {arrow_dir}")
    tables = []
    for f in files:
        with open(f, "rb") as src:
            reader = pa.ipc.open_stream(src)
            tables.append(reader.read_all())
    return pa.concat_tables(tables)


def load_model(device):
    """augreg2 safetensors 로드 (세미 세팅)"""
    model = timm.create_model("vit_base_patch16_224", pretrained=False)
    cache_path = os.path.expanduser(
        "~/.cache/huggingface/hub/"
        "models--timm--vit_base_patch16_224.augreg2_in21k_ft_in1k/"
        "snapshots/063c6c38a5d8510b2e57df480445e94b231dad2c/model.safetensors"
    )
    state_dict = load_file(cache_path)
    model.load_state_dict(state_dict, strict=False)
    return model.to(device).eval()


# ── 전역 버퍼 ──────────────────────────────────────────────────────────────────
_stats_buffer = defaultdict(list)


def compute_salience(keys: torch.Tensor) -> torch.Tensor:
    """
    keys: (B, N, d)
    return: (B, N) min-max normalized salience
    """
    affinity = torch.bmm(keys, keys.transpose(1, 2))   # (B, N, N)
    hat_A    = affinity.softmax(dim=-1)
    salience = hat_A.sum(dim=1)                         # (B, N)
    s_min = salience.min(dim=-1, keepdim=True).values
    s_max = salience.max(dim=-1, keepdim=True).values
    return (salience - s_min) / (s_max - s_min + 1e-6)


def _bipartite_merge_with_logging(x: torch.Tensor,
                                   r: int,
                                   salience: torch.Tensor):
    """
    merge되는 pair의 cosine similarity와 salience asymmetry를 버퍼에 기록.
    (반환값 없음 — 통계 수집 목적)
    """
    B, N, d = x.shape
    half = N // 2
    a, b = x[:, :half], x[:, half:]
    sa   = salience[:, :half]
    sb   = salience[:, half:]

    a_norm = F.normalize(a, dim=-1)
    b_norm = F.normalize(b, dim=-1)
    sim    = torch.bmm(a_norm, b_norm.transpose(1, 2))  # (B, half, half)

    best_sim, best_idx = sim.max(dim=-1)                # (B, half)
    top_sim, top_a_idx = best_sim.topk(min(r, half), dim=-1)   # (B, r)
    top_b_idx = best_idx.gather(1, top_a_idx)           # (B, r)

    # (1) cosine similarity
    _stats_buffer['cos'].extend(
        top_sim.detach().cpu().float().view(-1).tolist()
    )

    # (2) salience asymmetry: (s_i - s_j)²
    s_a = sa.gather(1, top_a_idx)
    s_b = sb.gather(1, top_b_idx)
    asym = (s_a - s_b) ** 2          # ← .cpu() 제거, GPU에 유지

    # (3) Proposition 1 leading term: ||x_i - x_j||² · (s_i-s_j)² / (s_i+s_j)²
    sq_dist = ((a.gather(1, top_a_idx.unsqueeze(-1).expand(-1,-1,d)) -
                b.gather(1, top_b_idx.unsqueeze(-1).expand(-1,-1,d))) ** 2).sum(-1)
    denom   = ((s_a + s_b) ** 2).clamp(min=1e-8)
    leading = (sq_dist * asym / denom).detach().cpu().float()  # 곱한 뒤에 CPU로
    _stats_buffer['leading'].extend(leading.view(-1).tolist())

    # 버퍼 저장은 마지막에 cpu로
    _stats_buffer['asym'].extend(asym.detach().cpu().float().view(-1).tolist())  # ← 여기서 cpu


def build_hook(r_fixed: int):
    """각 ViT Block의 forward_pre_hook — key 벡터 추출 후 통계 수집."""
    def hook_fn(module, input):
        x = input[0].detach()          # (B, N, d)
        B, N, d = x.shape
        if N < 2 or r_fixed < 1:
            return
        try:
            qkv_w = module.attn.qkv.weight   # (3d, d)
            qkv_b = module.attn.qkv.bias     # (3d,)
            qkv   = F.linear(x, qkv_w, qkv_b)   # (B, N, 3d)
            _, keys, _ = qkv.chunk(3, dim=-1)    # (B, N, d)
        except AttributeError:
            return
        salience = compute_salience(keys)
        _bipartite_merge_with_logging(keys, r=r_fixed, salience=salience)
    return hook_fn


def run_for_r(model, loader, r_fixed: int,
              num_batches: int, device) -> tuple:
    _stats_buffer.clear()

    hooks = [
        block.register_forward_pre_hook(build_hook(r_fixed))
        for block in model.blocks
    ]

    model.eval()
    with torch.no_grad():
        for i, (images, _) in enumerate(loader):
            if i >= num_batches:
                break
            model(images.to(device))

    for h in hooks:
        h.remove()

    return (float(np.mean(_stats_buffer['cos'])),
            float(np.mean(_stats_buffer['asym'])),
            float(np.mean(_stats_buffer['leading'])))  # ← 추가


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir',    type=str,   required=True)
    parser.add_argument('--batch_size',  type=int,   default=64)
    parser.add_argument('--num_batches', type=int,   default=20)
    parser.add_argument('--r_values',    type=int,   nargs='+', default=[3,4,5,6,7,8])
    parser.add_argument('--output_json', type=str,   default='merge_stats.json')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    print("모델 로드 중...")
    model = load_model(device)
    print("완료\n")

    print("Validation set 로드 중...")
    val_table = load_arrow(args.data_dir, "validation")
    val_ds    = ArrowDataset(val_table, transform=TRANSFORM)
    loader    = DataLoader(val_ds, batch_size=args.batch_size,
                           shuffle=False, num_workers=4, pin_memory=True)
    print(f"Val: {len(val_ds):,}장  |  수집: {args.num_batches}배치 × {args.batch_size} = {args.num_batches * args.batch_size}장\n")

    results = {}
    print(f"{'r':>4} | {'⟨cos⟩':>8} | {'⟨(sᵢ-sⱼ)²⟩':>14} | {'⟨Prop.1⟩':>12}")
    print("-" * 46)
    for r in args.r_values:
        cos_mean, asym_mean, lead_mean = run_for_r(model, loader, r, args.num_batches, device)  # ← 수정
        results[r] = {'cos_mean': cos_mean, 'asym_mean': asym_mean, 'leading_mean': lead_mean}
        print(f"{r:>4} | {cos_mean:>8.4f} | {asym_mean:>14.6f} | {lead_mean:>12.6f}")

    with open(args.output_json, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n저장: {args.output_json}")

    r_min, r_max = min(args.r_values), max(args.r_values)
    cos_drop   = (results[r_min]['cos_mean'] - results[r_max]['cos_mean']) / results[r_min]['cos_mean'] * 100
    asym_mult  = results[r_max]['asym_mean']     / results[r_min]['asym_mean']
    lead_mult  = results[r_max]['leading_mean']  / results[r_min]['leading_mean']  # ← 추가

    print(f"\n논문 수치:")
    print(f"  ⟨cos⟩       감소율 : {cos_drop:.1f}%")
    print(f"  ⟨(sᵢ-sⱼ)²⟩ 변화   : {asym_mult:.2f}×")
    print(f"  ⟨Prop.1⟩    변화   : {lead_mult:.2f}×  ← 이게 증가해야 내러티브 성립")

if __name__ == '__main__':
    main()