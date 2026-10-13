"""
check_dataset.py
----------------
ImageNet-1k 데이터셋 다운로드 확인 스크립트.

사용법:
  python check_dataset.py --data_dir ./data
"""

import os
import argparse
import torch
from torchvision import transforms
from torch.utils.data import DataLoader


def check_dataset(data_dir: str):
    from datasets import load_dataset

    print("=" * 50)
    print("ImageNet-1k 데이터셋 확인")
    print("=" * 50)

    # ── 1. 로드 시도 ──────────────────────────
    print("\n[1/4] 데이터셋 로드 중...")
    try:
        hf_data = load_dataset(
            "ILSVRC/imagenet-1k",
            cache_dir=data_dir,
            trust_remote_code=True,
        )
        print("  ✅ 로드 성공")
    except Exception as e:
        print(f"  ❌ 로드 실패: {e}")
        return

    # ── 2. 분할 확인 ──────────────────────────
    print("\n[2/4] 데이터 분할 확인...")
    for split in ["train", "validation", "test"]:
        if split in hf_data:
            n = len(hf_data[split])
            print(f"  ✅ {split:12s}: {n:>10,}장")
        else:
            print(f"  ❌ {split:12s}: 없음")

    # ── 3. 샘플 이미지 확인 ───────────────────
    print("\n[3/4] 샘플 이미지 확인...")
    try:
        sample = hf_data["train"][0]
        img = sample["image"]
        label = sample["label"]
        print(f"  ✅ 이미지 크기:  {img.size} (W×H)")
        print(f"  ✅ 이미지 모드:  {img.mode}")
        print(f"  ✅ 레이블:       {label}")
        print(f"  ✅ 클래스 수:    {hf_data['train'].features['label'].num_classes}")
    except Exception as e:
        print(f"  ❌ 샘플 확인 실패: {e}")
        return

    # ── 4. DataLoader 테스트 ──────────────────
    print("\n[4/4] DataLoader 테스트 (배치 1개)...")
    try:
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])

        class SimpleDataset(torch.utils.data.Dataset):
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

        # train에서 10개만 테스트
        subset = torch.utils.data.Subset(
            SimpleDataset(hf_data["train"], transform),
            list(range(10))
        )
        loader = DataLoader(subset, batch_size=4, num_workers=0)
        imgs, labels = next(iter(loader))

        print(f"  ✅ 배치 shape:   {imgs.shape}")
        print(f"  ✅ 픽셀 범위:    min={imgs.min():.3f}, max={imgs.max():.3f}")
        print(f"  ✅ 레이블:       {labels.tolist()}")

    except Exception as e:
        print(f"  ❌ DataLoader 테스트 실패: {e}")
        return

    # ── 최종 요약 ─────────────────────────────
    print("\n" + "=" * 50)
    print("✅ 데이터셋 확인 완료! 학습 준비 OK")
    print("=" * 50)

    train_size = len(hf_data["train"])
    val_size   = len(hf_data["validation"])
    print(f"\n  Train:      {train_size:>10,}장")
    print(f"  Validation: {val_size:>10,}장")
    print(f"  클래스 수:  {hf_data['train'].features['label'].num_classes:>10,}개")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./data")
    args = parser.parse_args()
    check_dataset(args.data_dir)