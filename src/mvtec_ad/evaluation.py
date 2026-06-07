"""Reusable evaluation loops for both anomaly-detection approaches."""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from mvtec_ad.metrics import EvaluationResult, evaluate_predictions
from mvtec_ad.models.autoencoder import ConvAutoencoder
from mvtec_ad.models.feature_knn import ResNetPatchKNNDetector


@torch.no_grad()
def evaluate_autoencoder(
    model: ConvAutoencoder,
    test_loader: DataLoader,
    device: torch.device,
    pixel_threshold: float | None = None,
):

    from pathlib import Path

    from mvtec_ad.visualization import plot_anomaly_result

    model.to(device).eval()

    scores, labels, masks, maps = [], [], [], []

    output_dir = Path("outputs/visualizations")
    output_dir.mkdir(parents=True, exist_ok=True)

    sample_index = 0

    for batch in tqdm(test_loader, desc="Evaluate AE", leave=False):

        images = batch["image"].to(device, non_blocking=True)

        anomaly_maps = model.anomaly_map(images).cpu()

        image_scores = anomaly_maps.flatten(start_dim=1).amax(dim=1)

        scores.append(image_scores.numpy())
        labels.append(batch["label"].numpy())
        masks.append(batch["mask"].numpy())
        maps.append(anomaly_maps.numpy())

        predicted_masks = anomaly_maps > 0.5

        for i in range(len(images)):

            plot_anomaly_result(
                image=images[i].cpu(),
                ground_truth_mask=batch["mask"][i],
                anomaly_map=anomaly_maps[i],
                predicted_mask=predicted_masks[i],
                title=f"Label={batch['label'][i].item()}",
                save_path=output_dir / f"sample_{sample_index}.png",
            )

            sample_index += 1

    predictions = _stack_predictions(scores, labels, masks, maps)

    result = evaluate_predictions(
        pixel_threshold=pixel_threshold,
        **predictions,
    )

    return result, predictions


def evaluate_feature_detector(
    detector: ResNetPatchKNNDetector,
    test_loader: DataLoader,
    device: torch.device,
    pixel_threshold: float | None = None,
) -> tuple[EvaluationResult, dict[str, np.ndarray]]:
    """Evaluate the frozen ResNet + KNN detector."""

    scores, labels, masks, maps = [], [], [], []
    for batch in tqdm(test_loader, desc="Evaluate ResNet+KNN", leave=False):
        image_scores, anomaly_maps = detector.predict_batch(batch["image"], device=device)
        scores.append(image_scores)
        labels.append(batch["label"].numpy())
        masks.append(batch["mask"].numpy())
        maps.append(anomaly_maps.numpy())

    predictions = _stack_predictions(scores, labels, masks, maps)
    result = evaluate_predictions(pixel_threshold=pixel_threshold, **predictions)
    return result, predictions


def _stack_predictions(
    scores: list[np.ndarray],
    labels: list[np.ndarray],
    masks: list[np.ndarray],
    maps: list[np.ndarray],
) -> dict[str, np.ndarray]:
    return {
        "image_scores": np.concatenate(scores).astype(np.float64),
        "image_labels": np.concatenate(labels).astype(int),
        "masks": np.concatenate(masks).astype(np.float32),
        "anomaly_maps": np.concatenate(maps).astype(np.float32),
    }
