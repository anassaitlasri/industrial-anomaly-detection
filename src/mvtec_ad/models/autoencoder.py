"""Improved convolutional autoencoder for industrial anomaly detection."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


class ConvBlock(nn.Module):

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class ConvAutoencoder(nn.Module):

    def __init__(self):
        super().__init__()

        self.enc1 = ConvBlock(3, 32)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = ConvBlock(32, 64)
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = ConvBlock(64, 128)
        self.pool3 = nn.MaxPool2d(2)

        self.bottleneck = ConvBlock(128, 256)

        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec3 = ConvBlock(256, 128)

        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = ConvBlock(128, 64)

        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec1 = ConvBlock(64, 32)

        self.final = nn.Conv2d(32, 3, kernel_size=1)

    def forward(self, x):

        e1 = self.enc1(x)
        p1 = self.pool1(e1)

        e2 = self.enc2(p1)
        p2 = self.pool2(e2)

        e3 = self.enc3(p2)
        p3 = self.pool3(e3)

        b = self.bottleneck(p3)

        u3 = self.up3(b)
        d3 = self.dec3(torch.cat([u3, e3], dim=1))

        u2 = self.up2(d3)
        d2 = self.dec2(torch.cat([u2, e2], dim=1))

        u1 = self.up1(d2)
        d1 = self.dec1(torch.cat([u1, e1], dim=1))

        return self.final(d1)

    @torch.no_grad()
    def anomaly_map(self, x: torch.Tensor) -> torch.Tensor:

        reconstruction = self.forward(x)

        error = torch.mean(
            (x - reconstruction) ** 2,
            dim=1,
            keepdim=True,
        )

        error = F.avg_pool2d(error, kernel_size=21, stride=1, padding=10)

        flat = error.flatten(start_dim=1)

        min_values = flat.min(dim=1).values[:, None, None, None]
        max_values = flat.max(dim=1).values[:, None, None, None]

        return (error - min_values) / (max_values - min_values + 1e-8)


def train_autoencoder(
    model: ConvAutoencoder,
    train_loader: DataLoader,
    device: torch.device,
    epochs: int = 100,
    learning_rate: float = 1e-4,
) -> list[float]:

    model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=1e-5,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs,
    )

    criterion = nn.L1Loss()

    history = []

    for epoch in range(epochs):

        model.train()

        running_loss = 0.0

        progress = tqdm(
            train_loader,
            desc=f"AE epoch {epoch + 1}/{epochs}",
            leave=False,
        )

        for batch in progress:

            images = batch["image"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            reconstructions = model(images)

            loss = criterion(reconstructions, images)

            loss.backward()

            optimizer.step()

            running_loss += loss.item() * images.size(0)

            progress.set_postfix(loss=loss.item())

        scheduler.step()

        epoch_loss = running_loss / len(train_loader.dataset)

        history.append(epoch_loss)

    return history