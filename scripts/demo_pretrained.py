from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a YOLO11 pretrained-model demo."
    )
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Image, folder, video, camera index, or stream address.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolo11n.pt",
        help="YOLO model weight file.",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.25,
        help="Confidence threshold.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help='Use "cpu" or GPU index such as "0".',
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display prediction window.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not 0.0 <= args.confidence <= 1.0:
        raise ValueError("confidence must be between 0 and 1.")

    model = YOLO(args.model)

    results = model.predict(
        source=args.source,
        conf=args.confidence,
        device=args.device,
        save=True,
        show=args.show,
        project=str(PROJECT_ROOT / "demo_outputs"),
        name="pretrained_detection",
        exist_ok=True,
    )

    print(f"Processed {len(results)} input item(s).")
    print(
        "Results saved to:",
        PROJECT_ROOT / "demo_outputs" / "pretrained_detection",
    )


if __name__ == "__main__":
    main()