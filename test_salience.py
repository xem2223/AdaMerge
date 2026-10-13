"""
test_salience.py
----------------
실제 ViT feature로 salience 동작 확인.

사용법:
  python test_salience.py
  python test_salience.py --image_path /path/to/image.jpg  (직접 이미지 지정)
"""

import torch
import timm
import argparse
import numpy as np
from torchvision import transforms
from PIL import Image

from salience import compute_token_salience


def make_synthetic_image():
    """
    외부 이미지 없을 때 사용하는 합성 이미지.
    중앙에 밝은 오브젝트, 주변은 어두운 배경 → 실제 이미지와 유사한 구조.
    """
    img = np.zeros((224, 224, 3), dtype=np.uint8) + 30  # 어두운 배경

    # 중앙에 밝은 오브젝트
    cx, cy = 112, 112
    for i in range(224):
        for j in range(224):
            dist = ((i - cx)**2 + (j - cy)**2) ** 0.5
            if dist < 60:
                intensity = int(200 * (1 - dist/60))
                img[i, j] = [intensity + 30, intensity // 2 + 20, 30]

    return Image.fromarray(img)


def test_with_real_vit(image_path=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── 이미지 로드 ───────────────────────────
    if image_path:
        print(f"이미지 로드: {image_path}")
        img = Image.open(image_path).convert("RGB")
    else:
        print("합성 이미지 사용 (중앙 오브젝트 + 어두운 배경)")
        img = make_synthetic_image()

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    x = transform(img).unsqueeze(0).to(device)  # [1, 3, 224, 224]
    print(f"입력 shape: {x.shape}\n")

    # ── ViT 로드 + Hook ────────────────────────
    print("ViT-B/16 pretrained 로드 중...")
    model = timm.create_model("vit_base_patch16_224", pretrained=True)
    model = model.to(device)
    model.eval()

    layer_features = {}

    def make_hook(l):
        def hook(module, input, output):
            layer_features[l] = output.detach()
        return hook

    hooks = []
    for l, block in enumerate(model.blocks):
        hooks.append(block.register_forward_hook(make_hook(l)))

    with torch.no_grad():
        _ = model(x)

    for h in hooks:
        h.remove()

    # ── 레이어별 Salience 통계 ─────────────────
    print("=" * 58)
    print("레이어별 Salience 통계 (실제 ViT feature)")
    print("=" * 58)
    print(f"{'Layer':>6} {'min':>8} {'max':>8} {'mean':>8} {'std':>8}")
    print("-" * 58)

    for l in range(12):
        feat  = layer_features[l]       # [1, 197, 768]
        patch = feat[:, 1:, :]          # [1, 196, 768]
        S     = compute_token_salience(patch)

        print(f"{l:>6} {S.min():>8.4f} {S.max():>8.4f} "
              f"{S.mean():>8.4f} {S.std():>8.4f}")

    # ── 레이어 6 상세 분석 ─────────────────────
    print("\n=== 레이어 6 상세 분석 (중간 레이어) ===")
    S6 = compute_token_salience(layer_features[6][:, 1:, :])[0]  # [196]

    top5 = S6.topk(10).indices.tolist()
    bot5 = S6.topk(10, largest=False).indices.tolist()

    print(f"상위 10 토큰 (중요한 영역):")
    for idx in top5:
        row, col = idx // 14, idx % 14
        print(f"  토큰 {idx:3d} → grid ({row:2d}, {col:2d}) | salience={S6[idx]:.4f}")

    print(f"\n하위 10 토큰 (덜 중요한 영역):")
    for idx in bot5:
        row, col = idx // 14, idx % 14
        print(f"  토큰 {idx:3d} → grid ({row:2d}, {col:2d}) | salience={S6[idx]:.4f}")

    print(f"\n※ 14×14 grid 기준, (0,0)=좌상단 (13,13)=우하단")

    # ── 중앙 vs 주변 salience 비교 ────────────
    print("\n=== 중앙 vs 주변 Salience 비교 ===")
    # 중앙 4×4 토큰 (grid 5~8행, 5~8열)
    center_idx = [r*14 + c for r in range(5, 9) for c in range(5, 9)]
    # 가장자리 토큰 (0행, 13행, 0열, 13열)
    edge_idx = (
        list(range(0, 14)) +          # 0행
        list(range(182, 196)) +       # 13행
        list(range(0, 196, 14)) +     # 0열
        list(range(13, 196, 14))      # 13열
    )
    edge_idx = list(set(edge_idx))

    center_sal = S6[center_idx].mean().item()
    edge_sal   = S6[edge_idx].mean().item()

    print(f"중앙 토큰 평균 salience: {center_sal:.4f}")
    print(f"가장자리 토큰 평균 salience: {edge_sal:.4f}")

    if center_sal > edge_sal:
        print("✅ 중앙(오브젝트)이 가장자리(배경)보다 높은 salience → 정상 동작!")
    else:
        print("⚠️  중앙과 가장자리 차이 미미 → 이미지에 따라 다를 수 있음")

    print("\n✅ salience.py 실제 ViT feature 테스트 완료!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", type=str, default=None,
                        help="테스트할 이미지 경로 (없으면 합성 이미지 사용)")
    args = parser.parse_args()
    test_with_real_vit(args.image_path)