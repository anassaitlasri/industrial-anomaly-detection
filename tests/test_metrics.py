import numpy as np

from mvtec_ad.metrics import best_f1_threshold, segmentation_metrics


def test_best_f1_threshold_separates_simple_scores():
    labels = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.8, 0.9])

    f1, threshold = best_f1_threshold(labels, scores)

    assert f1 > 0.99
    assert 0.2 <= threshold <= 0.8


def test_segmentation_metrics_perfect_prediction():
    masks = np.array([[[[0, 1], [0, 1]]]], dtype=np.float32)
    anomaly_maps = np.array([[[[0.1, 0.9], [0.2, 0.8]]]], dtype=np.float32)

    dice, iou, threshold = segmentation_metrics(masks, anomaly_maps, threshold=0.5)

    assert dice > 0.99
    assert iou > 0.99
    assert threshold == 0.5
