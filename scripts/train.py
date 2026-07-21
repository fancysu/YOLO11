from __future__ import annotations

import argparse
from pathlib import Path

import torch
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a YOLO11 industrial vision model."
    )
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to the dataset YAML file.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolo11n.pt",
        help="Initial pretrained model.",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help='GPU index such as "0", or "cpu".',
    )
    parser.add_argument(
        "--name",
        type=str,
        default="industrial_yolo11",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    data_path = Path(args.data)
    if not data_path.is_absolute():
        data_path = PROJECT_ROOT / data_path

    if not data_path.exists():
        raise FileNotFoundError(f"Dataset YAML does not exist: {data_path}")

    device = args.device
    if device is None:
        device = "0" if torch.cuda.is_available() else "cpu"

    print(f"Dataset: {data_path}")
    print(f"Device : {device}")
    print(f"Model  : {args.model}")

    model = YOLO(args.model)

    model.train(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=device,
        project=str(PROJECT_ROOT / "runs"),
        name=args.name,
        pretrained=True,
        patience=30,
        save=True,
        plots=True,
        exist_ok=True,
    )


if __name__ == "__main__":
    main()