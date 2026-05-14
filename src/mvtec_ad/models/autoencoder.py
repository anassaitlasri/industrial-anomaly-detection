"""Convolutional autoencoder baseline for reconstruction-error anomaly maps."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


class ConvAutoencoder(nn.Module):
    """Compact convolutional autoencoder trained only on healthy transistors."""

    def __init__(self, latent_channels: int = 256) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            self._block(3, 32),
            self._block(32, 64),
            self._block(64, 128),
            self._block(128, latent_channels),
        )
        self.decoder = nn.Sequential(
            self._up_block(latent_channels, 128),
            self._up_block(128, 64),
            self._up_block(64, 32),
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),
        )

    @staticmethod
    def _block(in_channels: int, out_channels: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

    @staticmethod
    def _up_block(in_channels: int, out_channels: int) -> nn.Sequential:
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))

    @torch.no_grad()
    def anomaly_map(self, x: torch.Tensor) -> torch.Tensor:
        """Return a normalized pixel-wise reconstruction-error map in ``[0, 1]``."""

        reconstruction = self.forward(x)
        reconstruction = F.interpolate(
            reconstruction,
            size=x.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        error = torch.mean((x - reconstruction) ** 2, dim=1, keepdim=True)
        flat = error.flatten(start_dim=1)
        min_values = flat.min(dim=1).values[:, None, None, None]
        max_values = flat.max(dim=1).values[:, None, None, None]
        return (error - min_values) / (max_values - min_values + 1e-8)


def train_autoencoder(
    model: ConvAutoencoder,
    train_loader: DataLoader,
    device: torch.device,
    epochs: int = 25,
    learning_rate: float = 1e-3,
) -> list[float]:
    """Train the autoencoder and return epoch-level reconstruction losses."""

    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    criterion = nn.MSELoss()
    history: list[float] = []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        progress = tqdm(train_loader, desc=f"AE epoch {epoch + 1}/{epochs}", leave=False)
        for batch in progress:
            images = batch["image"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            reconstructions = model(images)
            loss = criterion(reconstructions, images)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
            progress.set_postfix(loss=loss.item())

        epoch_loss = running_loss / len(train_loader.dataset)
        history.append(epoch_loss)
    return history
