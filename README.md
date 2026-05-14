# MVTec AD Transistor — Industrial Anomaly Detection

Production-style project for a technical interview around semiconductor visual inspection, low-defect regimes, and yield protection. The code targets the **MVTec AD `transistor` category** and implements two complementary unsupervised anomaly-detection approaches in PyTorch and scikit-learn.

## Why this project matters for semiconductor manufacturing

In wafer/chip inspection, defective examples are scarce, labels are expensive, and the business goal is not only classification accuracy: it is fast detection of process drift, actionable defect localization, and improved yield. This project therefore trains only on healthy transistors and evaluates both:

- **Image-level detection**: should the part be escalated or rejected?
- **Pixel-level localization**: where is the suspicious region for process engineers?

## Project structure

```text
src/mvtec_ad/
  data.py                    # MVTec train/test dataset and dataloaders
  models/autoencoder.py      # Model A: convolutional reconstruction baseline
  models/feature_knn.py      # Model B: frozen ResNet18 + KNN patch scoring
  evaluation.py              # Shared evaluation loops
  metrics.py                 # AUROC, F1, Dice, IoU
  visualization.py           # Original image / GT mask / heatmap plots
  cli.py                     # Reproducible command-line interface
scripts/
  run_autoencoder.py
  run_feature_knn.py
tests/
  test_metrics.py
```

## Dataset layout

Download MVTec AD and keep the standard folder structure:

```text
/path/to/mvtec/transistor/
  train/good/*.png
  test/good/*.png
  test/<defect_type>/*.png
  ground_truth/<defect_type>/*_mask.png
```

The loader accepts either `/path/to/mvtec` or `/path/to/mvtec/transistor`.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Run Model A — Convolutional Autoencoder

```bash
python -m mvtec_ad.cli autoencoder \
  --data-root /path/to/mvtec \
  --epochs 25 \
  --batch-size 16 \
  --image-size 256
```

**Principle:** the autoencoder sees only normal transistors during training. At test time, defects are expected to reconstruct poorly. The pixel-wise reconstruction error becomes an anomaly heatmap.

## Run Model B — Frozen ResNet18 + KNN patch detector

```bash
python -m mvtec_ad.cli feature-knn \
  --data-root /path/to/mvtec \
  --batch-size 16 \
  --image-size 256 \
  --n-neighbors 5
```

**Principle:** a pretrained ResNet18 converts images into local semantic patch embeddings. KNN learns the manifold of normal patches. At inference, patches far from the normal memory bank are anomalous; upsampled patch distances become a localization heatmap.

## Metrics reported

The CLI prints a JSON object with:

- `image_auroc`: ranking quality for normal vs defective test images.
- `image_f1`: operating-point quality after threshold selection.
- `image_threshold`: selected image decision threshold.
- `pixel_dice`: overlap between predicted defect mask and ground truth.
- `pixel_iou`: stricter overlap metric for segmentation quality.
- `pixel_threshold`: selected heatmap-to-mask threshold.

## Visualization example

```python
from mvtec_ad.visualization import plot_anomaly_result

fig = plot_anomaly_result(
    image=batch["image"][0],
    ground_truth_mask=batch["mask"][0],
    anomaly_map=anomaly_maps[0],
    predicted_mask=anomaly_maps[0] > 0.5,
    title="Transistor anomaly localization",
    save_path="outputs/example.png",
)
```

## Executive interview pitch

### 1) Simple explanation of both approaches

**Model A — Autoencoder:** I train a compact neural network to compress and reconstruct only healthy transistor images. If a transistor contains an unusual defect, the model has not learned how to reproduce it correctly. The reconstruction error highlights suspicious pixels and provides both an image anomaly score and a defect heatmap.

**Model B — ResNet18 + KNN:** Instead of learning visual features from scratch, I reuse a robust ImageNet-pretrained backbone as a feature extractor. I store representative patch embeddings from healthy transistors. During inspection, I compare each new patch to its nearest healthy neighbors. If a local region is visually far from the normal population, it is flagged as anomalous.

### 2) Strategic comparison: why industry often prefers Model B

- **Faster time-to-value:** Model B avoids long training cycles and works well in low-defect regimes because the heavy visual representation is already learned.
- **Lower compute cost:** Only frozen inference plus a classical KNN index is needed; no backpropagation is required for the main detector.
- **More MLOps-friendly:** The backbone can be versioned, frozen, validated, and monitored separately from the classical detector. Updating the normal memory bank is simpler than retraining an end-to-end network.
- **Better ROI:** In factories, the fastest deployable system that reliably reduces escapes and false alarms often wins over a theoretically elegant model requiring more data, tuning, and GPU time.
- **Interpretability for engineers:** Patch-distance heatmaps are easy to explain: “this area does not look like previously accepted healthy parts.”

The autoencoder remains useful as a transparent baseline, a sanity check, and a demonstration that reconstruction-based unsupervised learning can localize defects. But for a production pilot, pretrained features plus a lightweight detector usually provide a better balance of performance, speed, maintainability, and risk.

### 3) How to sell this to STMicroelectronics

I would position the project as a **yield-protection and process-monitoring tool** rather than a pure computer-vision demo:

- **Yield improvement:** Early anomaly detection can reduce defective dies progressing downstream, saving expensive later-stage testing and packaging capacity.
- **Low-label compatibility:** The method trains on normal production images, which matches semiconductor reality: many defect classes are rare, evolving, or discovered after process changes.
- **Root-cause acceleration:** Pixel-level heatmaps help process and equipment engineers quickly localize suspicious regions and correlate them with tool settings, lot history, recipes, or metrology signals.
- **Fab-ready deployment path:** Model B can be packaged as a reproducible inference service, tracked with model/data versions, monitored for drift, and periodically refreshed with newly validated normal samples.
- **Business framing:** The value is measured through fewer false escapes, fewer unnecessary manual reviews, faster excursion detection, improved line stability, and ultimately better yield.

A strong interview sentence: **“I designed the solution around the semiconductor constraint that defects are rare and labels are costly. The system learns the visual distribution of known-good transistors, flags deviations, localizes them for engineers, and can be integrated into an MLOps loop to protect yield as the process drifts.”**
