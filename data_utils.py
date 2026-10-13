from typing import Optional, Sequence

import torch
from torch.utils.data import DataLoader
from torchvision import transforms


IMAGENET_TRANSFORM = transforms.Compose([
    transforms.Resize(249, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])


class HFDatasetWrapper(torch.utils.data.Dataset):
    def __init__(self, hf_dataset, indices: Optional[Sequence[int]] = None, transform=None):
        self.dataset = hf_dataset
        self.indices = list(range(len(hf_dataset))) if indices is None else list(indices)
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        item = self.dataset[int(self.indices[idx])]
        image = item["image"].convert("RGB")
        label = item["label"]
        if self.transform is not None:
            image = self.transform(image)
        return image, label


def build_imagenet_loader(
    data_dir: str,
    split: str,
    batch_size: int,
    num_workers: int,
    indices: Optional[Sequence[int]] = None,
) -> DataLoader:
    """Build an ImageNet-1k DataLoader via HuggingFace datasets.

    Users must have access to the ImageNet-1k dataset and configure the
    HuggingFace cache/authentication as required by the dataset provider.
    """
    from datasets import load_dataset

    dataset = load_dataset("ILSVRC/imagenet-1k", cache_dir=data_dir)[split]
    wrapped = HFDatasetWrapper(dataset, indices=indices, transform=IMAGENET_TRANSFORM)
    return DataLoader(
        wrapped,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
