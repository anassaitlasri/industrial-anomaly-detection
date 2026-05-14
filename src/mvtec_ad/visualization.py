"""Visualization helpers for anomaly inspection reports."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def denormalize_image(image: torch.Tensor) -> np.ndarray:
    """Convert an ImageNet-normalized tensor to a displayable HWC array."""

    image = image.detach().cpu() * IMAGENET_STD + IMAGENET_MEAN
    image = image.clamp(0, 1).permute(1, 2, 0).numpy()
    return image


def plot_anomaly_result(
    image: torch.Tensor | np.ndarray,
    ground_truth_mask: torch.Tensor | np.ndarray,
    anomaly_map: torch.Tensor | np.ndarray,
    predicted_mask: torch.Tensor | np.ndarray | None = None,
    title: str | None = None,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Display original image, ground truth, heatmap, and optional binary prediction."""

    image_array = denormalize_image(image) if isinstance(image, torch.Tensor) else image
    gt_array = _to_2d_array(ground_truth_mask)
    heatmap_array = _to_2d_array(anomaly_map)

    panels = [
        ("Original", image_array, None),
        ("Ground truth", gt_array, "gray"),
        ("Anomaly heatmap", heatmap_array, "magma"),
    ]
    if predicted_mask is not None:
        panels.append(("Predicted mask", _to_2d_array(predicted_mask), "gray"))

    fig, axes = plt.subplots(1, len(panels), figsize=(4 * len(panels), 4), constrained_layout=True)
    if len(panels) == 1:
        axes = [axes]
    for axis, (panel_title, panel_data, cmap) in zip(axes, panels, strict=True):
        axis.imshow(panel_data, cmap=cmap)
        axis.set_title(panel_title)
        axis.axis("off")
    if title:
        fig.suptitle(title)
    if save_path:
        fig.savefig(save_path, dpi=160, bbox_inches="tight")
    return fig


def _to_2d_array(value: torch.Tensor | np.ndarray) -> np.ndarray:
    array = value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else value
    array = np.squeeze(array)
    if array.ndim != 2:
        raise ValueError(f"Expected a 2D mask/heatmap after squeeze, got shape {array.shape}.")
    return array
