"""Dataset and dataloader utilities for MVTec AD transistor inspection.

Expected directory layout after downloading/extracting MVTec AD::

    <data_root>/transistor/
        train/good/*.png
        test/good/*.png
        test/<defect_type>/*.png
        ground_truth/<defect_type>/*_mask.png

The training split intentionally exposes only healthy images, matching the
industrial setting where defective wafers/chips are rare or not fully labeled.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

Split = Literal["train", "test"]


@dataclass(frozen=True)
class MVTecSample:
    """Metadata for one MVTec image."""

    image_path: Path
    mask_path: Path | None
    label: int
    defect_type: str


class MVTecTransistorDataset(Dataset):
    """Load MVTec AD Transistor images and pixel-level ground truth masks.

    Args:
        root: Path to either the MVTec root containing ``transistor`` or directly
            to the ``transistor`` category directory.
        split: ``train`` returns only normal images; ``test`` returns normal and
            anomalous images.
        image_size: Square resize used by both the image and mask transforms.
        image_transform: Optional override for image preprocessing.
        mask_transform: Optional override for mask preprocessing.
    """

    def __init__(
        self,
        root: str | Path,
        split: Split,
        image_size: int = 256,
        image_transform: Callable | None = None,
        mask_transform: Callable | None = None,
    ) -> None:
        self.category_dir = self._resolve_category_dir(Path(root))
        self.split = split
        self.image_size = image_size
        self.image_transform = image_transform or self.default_image_transform(image_size)
        self.mask_transform = mask_transform or self.default_mask_transform(image_size)
        self.samples = self._discover_samples()

        if not self.samples:
            raise FileNotFoundError(
                f"No samples found for split='{split}' under {self.category_dir}. "
                "Check that the MVTec AD Transistor dataset is extracted correctly."
            )

    @staticmethod
    def _resolve_category_dir(root: Path) -> Path:
        return root if root.name == "transistor" else root / "transistor"

    @staticmethod
    def default_image_transform(image_size: int) -> transforms.Compose:
        return transforms.Compose(
            [
                transforms.Resize((image_size, image_size), antialias=True),

                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.2),
                transforms.RandomRotation(degrees=5),

                transforms.ColorJitter(
                    brightness=0.1,
                    contrast=0.1,
                    saturation=0.1,
                    hue=0.02,
                ),

                transforms.ToTensor(),

                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                ),
            ]
        )

    @staticmethod
    def default_mask_transform(image_size: int) -> transforms.Compose:
        return transforms.Compose(
            [
                transforms.Resize(
                    (image_size, image_size),
                    interpolation=transforms.InterpolationMode.NEAREST,
                ),
                transforms.ToTensor(),
            ]
        )

    def _discover_samples(self) -> list[MVTecSample]:
        if self.split == "train":
            return [
                MVTecSample(image_path=path, mask_path=None, label=0, defect_type="good")
                for path in sorted((self.category_dir / "train" / "good").glob("*.png"))
            ]

        samples: list[MVTecSample] = []
        for defect_dir in sorted((self.category_dir / "test").iterdir()):
            if not defect_dir.is_dir():
                continue
            defect_type = defect_dir.name
            label = 0 if defect_type == "good" else 1
            for image_path in sorted(defect_dir.glob("*.png")):
                mask_path = None if label == 0 else self._mask_path_for(image_path, defect_type)
                samples.append(
                    MVTecSample(
                        image_path=image_path,
                        mask_path=mask_path,
                        label=label,
                        defect_type=defect_type,
                    )
                )
        return samples

    def _mask_path_for(self, image_path: Path, defect_type: str) -> Path:
        mask_dir = self.category_dir / "ground_truth" / defect_type
        candidates = [
            mask_dir / f"{image_path.stem}_mask.png",
            mask_dir / image_path.name,
            mask_dir / f"{image_path.stem}.png",
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        raise FileNotFoundError(f"Mask not found for anomalous image: {image_path}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        sample = self.samples[index]
        image = Image.open(sample.image_path).convert("RGB")
        image_tensor = self.image_transform(image)

        if sample.mask_path is None:
            mask_tensor = torch.zeros((1, self.image_size, self.image_size), dtype=torch.float32)
        else:
            mask = Image.open(sample.mask_path).convert("L")
            mask_tensor = (self.mask_transform(mask) > 0.5).float()

        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "label": torch.tensor(sample.label, dtype=torch.long),
            "image_path": str(sample.image_path),
            "defect_type": sample.defect_type,
        }


def build_dataloader(
    root: str | Path,
    split: Split,
    batch_size: int,
    image_size: int = 256,
    num_workers: int = 4,
    shuffle: bool | None = None,
) -> DataLoader:
    """Create a DataLoader with sensible defaults for train/test phases."""

    dataset = MVTecTransistorDataset(root=root, split=split, image_size=image_size)
    should_shuffle = split == "train" if shuffle is None else shuffle
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=should_shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
