from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import cv2
import pandas as pd
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect and count aluminium profile products."
    )
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--source", type=str, required=True)
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--device", type=str, default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    weights_path = Path(args.weights)
    source_path = Path(args.source)

    if not weights_path.is_absolute():
        weights_path = PROJECT_ROOT / weights_path
    if not source_path.is_absolute():
        source_path = PROJECT_ROOT / source_path

    if not weights_path.exists():
        raise FileNotFoundError(f"Weights not found: {weights_path}")
    if not source_path.exists():
        raise FileNotFoundError(f"Source image not found: {source_path}")

    model = YOLO(str(weights_path))

    results = model.predict(
        source=str(source_path),
        conf=args.conf,
        device=args.device,
        verbose=False,
    )

    result = results[0]
    counts: Counter[str] = Counter()

    if result.boxes is not None:
        for class_id in result.boxes.cls.cpu().tolist():
            class_name = result.names[int(class_id)]
            counts[class_name] += 1

    output_dir = PROJECT_ROOT / "demo_outputs" / "inventory_count"
    output_dir.mkdir(parents=True, exist_ok=True)

    annotated_image = result.plot()

    summary_lines = ["Inventory Summary"]
    for class_name, quantity in sorted(counts.items()):
        summary_lines.append(f"{class_name}: {quantity}")

    summary_text = "\n".join(summary_lines)
    print(summary_text)

    y = 35
    for line in summary_lines:
        cv2.putText(
            annotated_image,
            line,
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
        y += 32

    image_output = output_dir / f"{source_path.stem}_counted.jpg"
    cv2.imwrite(str(image_output), annotated_image)

    table = pd.DataFrame(
        [
            {"product_type": class_name, "quantity": quantity}
            for class_name, quantity in sorted(counts.items())
        ]
    )

    excel_output = output_dir / f"{source_path.stem}_inventory.xlsx"
    table.to_excel(excel_output, index=False)

    print(f"Annotated image: {image_output}")
    print(f"Inventory table : {excel_output}")


if __name__ == "__main__":
    main()

#counting运行：
#python scripts/validate.py --weights runs/inventory_yolo11n/weights/best.pt --data configs/inventory_data.yaml --device 0
#会输出：
# profile_type_a: 12
# profile_type_b: 8
# profile_type_c: 5
# 检测结果图片和库存统计excel