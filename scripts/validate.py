from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a trained YOLO model.")
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", type=str, default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    weights = Path(args.weights)
    data = Path(args.data)

    if not weights.is_absolute():
        weights = PROJECT_ROOT / weights
    if not data.is_absolute():
        data = PROJECT_ROOT / data

    if not weights.exists():
        raise FileNotFoundError(f"Weights not found: {weights}")
    if not data.exists():
        raise FileNotFoundError(f"Dataset YAML not found: {data}")

    model = YOLO(str(weights))

    metrics = model.val(
        data=str(data),
        imgsz=args.imgsz,
        device=args.device,
        project=str(PROJECT_ROOT / "runs"),
        name="validation",
    )

    print("Validation completed.")
    print(f"mAP50-95: {metrics.box.map:.4f}")
    print(f"mAP50   : {metrics.box.map50:.4f}")
    print(f"mAP75   : {metrics.box.map75:.4f}")


if __name__ == "__main__":
    main()

#验证代码，运行：
#python scripts/validate.py --weights runs/inventory_yolo11n/weights/best.pt --data configs/inventory_data.yaml --device 0