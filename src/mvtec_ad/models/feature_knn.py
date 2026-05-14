"""Pretrained ResNet18 feature extraction with KNN patch anomaly scoring.

This module implements an industry-friendly alternative to training a model from
scratch: use ImageNet-pretrained representations as a frozen visual backbone and
fit a small classical detector on normal embeddings. The detector returns both
image-level scores and coarse pixel-level anomaly maps by scoring local patches.
"""

from __future__ import annotations

import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision.models import ResNet18_Weights, resnet18
from tqdm.auto import tqdm


class ResNet18FeatureExtractor(nn.Module):
    """Frozen ResNet18 backbone exposing intermediate patch features."""

    def __init__(self) -> None:
        super().__init__()
        backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.eval()
        for parameter in self.parameters():
            parameter.requires_grad = False

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        return self.layer3(x)


class ResNetPatchKNNDetector:
    """KNN detector trained on normal ResNet patch embeddings.

    Args:
        n_neighbors: Number of nearest normal patches used for anomaly distance.
        max_memory_patches: Optional random memory-bank cap for faster inference.
        random_state: Deterministic seed for memory-bank subsampling.
    """

    def __init__(
        self,
        n_neighbors: int = 5,
        max_memory_patches: int = 50_000,
        random_state: int = 42,
    ) -> None:
        self.extractor = ResNet18FeatureExtractor()
        self.n_neighbors = n_neighbors
        self.max_memory_patches = max_memory_patches
        self.random_state = random_state
        self.knn = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean")
        self.is_fitted = False

    def fit(self, train_loader: DataLoader, device: torch.device) -> None:
        """Build the normal patch memory bank and fit the nearest-neighbor index."""

        self.extractor.to(device).eval()
        embeddings: list[np.ndarray] = []
        for batch in tqdm(train_loader, desc="Extract normal ResNet patches", leave=False):
            images = batch["image"].to(device, non_blocking=True)
            features = self.extractor(images)
            patches = self._features_to_patches(features)
            embeddings.append(patches.cpu().numpy())

        memory_bank = np.concatenate(embeddings, axis=0)
        if len(memory_bank) > self.max_memory_patches:
            rng = np.random.default_rng(self.random_state)
            indices = rng.choice(len(memory_bank), size=self.max_memory_patches, replace=False)
            memory_bank = memory_bank[indices]

        self.knn.fit(memory_bank)
        self.is_fitted = True

    @torch.no_grad()
    def predict_batch(
        self,
        images: torch.Tensor,
        device: torch.device,
    ) -> tuple[np.ndarray, torch.Tensor]:
        """Return image anomaly scores and normalized anomaly maps for a batch."""

        if not self.is_fitted:
            raise RuntimeError("ResNetPatchKNNDetector must be fitted before prediction.")

        self.extractor.to(device).eval()
        images = images.to(device, non_blocking=True)
        features = self.extractor(images)
        batch_size, _, height, width = features.shape
        patches = self._features_to_patches(features).cpu().numpy()
        distances, _ = self.knn.kneighbors(patches)
        patch_scores = distances.mean(axis=1).reshape(batch_size, height, width)
        maps = torch.from_numpy(patch_scores).float().unsqueeze(1).to(device)
        maps = F.interpolate(maps, size=images.shape[-2:], mode="bilinear", align_corners=False)
        maps = self._normalize_maps(maps)
        image_scores = maps.flatten(start_dim=1).amax(dim=1).cpu().numpy()
        return image_scores, maps.cpu()

    @staticmethod
    def _features_to_patches(features: torch.Tensor) -> torch.Tensor:
        patches = features.permute(0, 2, 3, 1).reshape(-1, features.shape[1])
        return F.normalize(patches, p=2, dim=1)

    @staticmethod
    def _normalize_maps(maps: torch.Tensor) -> torch.Tensor:
        flat = maps.flatten(start_dim=1)
        min_values = flat.min(dim=1).values[:, None, None, None]
        max_values = flat.max(dim=1).values[:, None, None, None]
        return (maps - min_values) / (max_values - min_values + 1e-8)
