from __future__ import annotations

import gc
import numpy as np
import torch
import faiss

# Restriction de Faiss à 1 seul thread pour éviter de saturer la RAM/CPU sous Windows
faiss.omp_set_num_threads(1)

from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from torchvision.models import (
    ResNet50_Weights,
    resnet50,
)

from tqdm.auto import tqdm


class PatchCoreFeatureExtractor(nn.Module):

    def __init__(self) -> None:
        super().__init__()

        backbone = resnet50(
            weights=ResNet50_Weights.IMAGENET1K_V2
        )

        self.stem = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
        )

        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3

        self.eval()

        for p in self.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)

        f2 = self.layer2(x)
        f3 = self.layer3(f2)

        # Upscale f3 vers f2 pour garder la haute résolution spatiale
        f3 = F.interpolate(
            f3,
            size=f2.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        features = torch.cat(
            [f2, f3],
            dim=1,
        )

        # Lissage spatial PatchCore pour inclure le contexte local
        features = F.avg_pool2d(
            features,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        return features


class PatchCoreDetector:

    def __init__(
        self,
        n_neighbors: int = 1, # PatchCore ne regarde que le voisin le plus proche
        max_memory_patches: int = 10000, # Plus besoin de 100k grâce au Coreset
        random_state: int = 42,
    ) -> None:

        self.extractor = PatchCoreFeatureExtractor()

        self.n_neighbors = n_neighbors
        self.max_memory_patches = max_memory_patches
        self.random_state = random_state

        self.index = None
        self.is_fitted = False

    def fit(
        self,
        train_loader: DataLoader,
        device: torch.device,
    ) -> None:

        self.extractor.to(device)
        self.extractor.eval()

        embeddings = []

        for batch in tqdm(
            train_loader,
            desc="Build PatchCore Memory Bank",
            leave=False,
        ):

            images = batch["image"].to(
                device,
                non_blocking=True,
            )

            features = self.extractor(images)
            patches = self._features_to_patches(features)

            embeddings.append(
                patches.cpu().numpy()
            )

            # Nettoyage de la RAM pour les machines avec mémoire limitée
            del images
            del features
            del patches
            gc.collect()

        memory_bank = np.concatenate(
            embeddings,
            axis=0,
        )

        print(
            f"Original memory bank size: {len(memory_bank):,}"
        )

        if len(memory_bank) > self.max_memory_patches:
            # Remplacement de l'échantillonnage aléatoire par le Greedy Coreset
            memory_bank = self._compute_coreset(
                memory_bank,
                target_size=self.max_memory_patches,
                random_state=self.random_state,
            )

        print(
            f"Coreset size: {len(memory_bank):,}"
        )

        # Initialisation et peuplement de l'index Faiss
        dimension = memory_bank.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        
        # Faiss exige le type float32
        memory_bank = memory_bank.astype(np.float32)
        
        self.index.add(memory_bank)
        self.is_fitted = True

    @torch.no_grad()
    def predict_batch(
        self,
        images: torch.Tensor,
        device: torch.device,
    ) -> tuple[np.ndarray, torch.Tensor]:

        if not self.is_fitted:
            raise RuntimeError(
                "PatchCoreDetector must be fitted first."
            )

        self.extractor.to(device)
        self.extractor.eval()

        images = images.to(
            device,
            non_blocking=True,
        )

        features = self.extractor(images)
        batch_size, _, h, w = features.shape

        patches = self._features_to_patches(
            features
        ).cpu().numpy()

        patches = patches.astype(np.float32)

        # Recherche rapide avec Faiss
        distances, _ = self.index.search(patches, k=self.n_neighbors)

        patch_scores = distances[:, 0]

        patch_scores = patch_scores.reshape(
            batch_size,
            h,
            w,
        )

        maps = torch.from_numpy(
            patch_scores
        ).float()

        maps = maps.unsqueeze(1).to(device)

        maps = F.interpolate(
            maps,
            size=images.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        # On extrait le score maximum de la carte d'anomalie brute
        image_scores = (
            maps.flatten(start_dim=1)
            .amax(dim=1)
            .cpu()
            .numpy()
        )
        
        return image_scores, maps.cpu()

    @staticmethod
    def _features_to_patches(
        features: torch.Tensor,
    ) -> torch.Tensor:

        patches = (
            features
            .permute(0, 2, 3, 1)
            .reshape(
                -1,
                features.shape[1],
            )
        )

        return F.normalize(
            patches,
            p=2,
            dim=1,
        )

    @staticmethod
    def _compute_coreset(
        memory_bank: np.ndarray,
        target_size: int,
        random_state: int = 42,
    ) -> np.ndarray:
        """
        Greedy K-Center Coreset algorithm.
        Sélectionne itérativement les patchs les plus diversifiés.
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tensor_bank = torch.from_numpy(memory_bank).to(device)
        
        # Réduction de dimension aléatoire (Johnson-Lindenstrauss) pour accélérer le calcul
        projection_dim = 128
        torch.manual_seed(random_state)
        random_matrix = torch.randn(tensor_bank.shape[1], projection_dim, device=device)
        projected_bank = torch.matmul(tensor_bank, random_matrix)
        
        num_patches = projected_bank.shape[0]
        min_distances = torch.full((num_patches,), float('inf'), device=device)
        selected_indices = []
        
        # Initialisation avec un patch aléatoire
        current_idx = int(torch.randint(0, num_patches, (1,), device=device).item())
        
        for _ in tqdm(range(target_size), desc="Greedy Coreset Sampling", leave=False):
            selected_indices.append(current_idx)
            
            current_point = projected_bank[current_idx].unsqueeze(0)
            
            # Calcul des distances Euclidiennes au carré
            distances = torch.sum((projected_bank - current_point) ** 2, dim=1)
            
            # Mise à jour des distances minimales au Coreset
            min_distances = torch.minimum(min_distances, distances)
            
            # Le prochain point sélectionné est celui qui est le plus éloigné
            current_idx = int(torch.argmax(min_distances).item())
            
        return memory_bank[selected_indices]