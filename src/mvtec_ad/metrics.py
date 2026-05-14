"""Industrial anomaly-detection metrics for image and pixel levels."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import f1_score, precision_recall_curve, roc_auc_score


@dataclass(frozen=True)
class EvaluationResult:
    """Aggregated evaluation metrics and selected operating thresholds."""

    image_auroc: float
    image_f1: float
    image_threshold: float
    pixel_dice: float
    pixel_iou: float
    pixel_threshold: float


def best_f1_threshold(labels: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    """Find the score threshold that maximizes image-level F1."""

    precision, recall, thresholds = precision_recall_curve(labels, scores)
    f1_values = 2 * precision * recall / (precision + recall + 1e-8)
    best_index = int(np.nanargmax(f1_values))
    if best_index >= len(thresholds):
        threshold = float(scores.max() + 1e-8)
    else:
        threshold = float(thresholds[best_index])
    return float(f1_values[best_index]), threshold


def segmentation_metrics(
    masks: np.ndarray,
    anomaly_maps: np.ndarray,
    threshold: float | None = None,
) -> tuple[float, float, float]:
    """Compute Dice and IoU for binary masks produced from anomaly maps."""

    masks_binary = masks.astype(bool)
    if threshold is None:
        threshold = float(np.percentile(anomaly_maps, 99.0))
    predictions = anomaly_maps >= threshold

    intersection = np.logical_and(predictions, masks_binary).sum(dtype=np.float64)
    predicted_area = predictions.sum(dtype=np.float64)
    target_area = masks_binary.sum(dtype=np.float64)
    union = np.logical_or(predictions, masks_binary).sum(dtype=np.float64)

    dice = (2.0 * intersection) / (predicted_area + target_area + 1e-8)
    iou = intersection / (union + 1e-8)
    return float(dice), float(iou), float(threshold)


def evaluate_predictions(
    image_labels: np.ndarray,
    image_scores: np.ndarray,
    masks: np.ndarray,
    anomaly_maps: np.ndarray,
    pixel_threshold: float | None = None,
) -> EvaluationResult:
    """Evaluate image-level detection and pixel-level segmentation quality."""

    image_auroc = float(roc_auc_score(image_labels, image_scores))
    image_f1, image_threshold = best_f1_threshold(image_labels, image_scores)
    binary_image_predictions = (image_scores >= image_threshold).astype(int)
    image_f1 = float(f1_score(image_labels, binary_image_predictions))
    pixel_dice, pixel_iou, resolved_pixel_threshold = segmentation_metrics(
        masks=masks,
        anomaly_maps=anomaly_maps,
        threshold=pixel_threshold,
    )
    return EvaluationResult(
        image_auroc=image_auroc,
        image_f1=image_f1,
        image_threshold=image_threshold,
        pixel_dice=pixel_dice,
        pixel_iou=pixel_iou,
        pixel_threshold=resolved_pixel_threshold,
    )
