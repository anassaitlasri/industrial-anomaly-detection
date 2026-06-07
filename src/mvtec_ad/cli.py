"""Command-line entry points for the MVTec transistor anomaly project."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from mvtec_ad.data import build_dataloader
from mvtec_ad.evaluation import evaluate_autoencoder, evaluate_feature_detector
from mvtec_ad.models import (
    ConvAutoencoder,
    ResNetPatchKNNDetector,
    PatchCoreDetector,
    train_autoencoder,
)

def _run_patchcore(
    args,
    train_loader,
    test_loader,
    device,
):
    detector = PatchCoreDetector(
        max_memory_patches=args.max_memory_patches,
    )

    detector.fit(
        train_loader,
        device=device,
    )

    result, _ = evaluate_feature_detector(
        detector,
        test_loader,
        device=device,
    )

    return result

def main() -> None:
    parser = argparse.ArgumentParser(description="MVTec AD Transistor anomaly detection")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ae_parser = subparsers.add_parser("autoencoder", help="Train and evaluate convolutional AE")
    _add_common_args(ae_parser)
    ae_parser.add_argument("--epochs", type=int, default=100)
    ae_parser.add_argument("--learning-rate", type=float, default=1e-4)
    ae_parser.add_argument("--checkpoint", type=Path, default=Path("outputs/autoencoder.pt"))

    knn_parser = subparsers.add_parser("feature-knn", help="Fit and evaluate frozen ResNet18 + KNN")
    _add_common_args(knn_parser)
    knn_parser.add_argument("--n-neighbors", type=int, default=5)
    knn_parser.add_argument("--max-memory-patches", type=int, default=50_000)

    patch_parser = subparsers.add_parser(
        "patchcore",
        help="PatchCore anomaly detector"
    )

    _add_common_args(patch_parser)

    patch_parser.add_argument(
        "--max-memory-patches",
        type=int,
        default=10000,
    )

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    train_loader = build_dataloader(
        args.data_root,
        split="train",
        batch_size=args.batch_size,
        image_size=args.image_size,
        num_workers=args.num_workers,
    )
    test_loader = build_dataloader(
        args.data_root,
        split="test",
        batch_size=args.batch_size,
        image_size=args.image_size,
        num_workers=args.num_workers,
        shuffle=False,
    )
    

    if args.command == "autoencoder":
        result = _run_autoencoder(
            args,
            train_loader,
            test_loader,
            device,
        )

    elif args.command == "feature-knn":
            result = _run_feature_knn(
                args,
                train_loader,
                test_loader,
                device,
            )

    else:
        result = _run_patchcore(
                args,
                train_loader,
                test_loader,
                device,
            )

    print(json.dumps(result.__dict__, indent=2, sort_keys=True))


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="Path to MVTec root or transistor dir",
    )
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--cpu", action="store_true", help="Force CPU even when CUDA is available")


def _run_autoencoder(args: argparse.Namespace, train_loader, test_loader, device: torch.device):
    model = ConvAutoencoder()
    train_autoencoder(
        model=model,
        train_loader=train_loader,
        device=device,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
    )
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.checkpoint)
    result, _ = evaluate_autoencoder(model, test_loader, device=device)
    return result


def _run_feature_knn(args: argparse.Namespace, train_loader, test_loader, device: torch.device):
    detector = ResNetPatchKNNDetector(
        n_neighbors=args.n_neighbors,
        max_memory_patches=args.max_memory_patches,
    )
    detector.fit(train_loader, device=device)
    result, _ = evaluate_feature_detector(detector, test_loader, device=device)
    return result


if __name__ == "__main__":
    main()
