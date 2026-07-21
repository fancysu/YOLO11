from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from skimage.color import deltaE_ciede2000, rgb2lab


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare the average colour of two product images."
    )
    parser.add_argument("--reference", type=str, required=True)
    parser.add_argument("--sample", type=str, required=True)
    parser.add_argument(
        "--threshold",
        type=float,
        default=2.0,
        help="Temporary warning threshold; must later be calibrated.",
    )
    return parser.parse_args()


def read_image(path: Path) -> np.ndarray:
    image_bgr = cv2.imread(str(path))

    if image_bgr is None:
        raise ValueError(f"Unable to read image: {path}")

    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def robust_average_lab(image_rgb: np.ndarray) -> np.ndarray:
    """
    Estimate the representative Lab colour.

    Very dark and nearly saturated pixels are excluded to reduce the influence
    of shadows and specular highlights. This remains a baseline rather than
    a production-grade colour measurement method.
    """

    image = image_rgb.astype(np.float32) / 255.0

    brightness = image.mean(axis=2)
    valid_mask = (brightness > 0.10) & (brightness < 0.90)

    if valid_mask.sum() < 100:
        raise ValueError(
            "Too few valid pixels. Check ROI, lighting, or image exposure."
        )

    lab_image = rgb2lab(image)
    valid_lab_pixels = lab_image[valid_mask]

    return np.median(valid_lab_pixels, axis=0)


def main() -> None:
    args = parse_args()

    reference_path = Path(args.reference)
    sample_path = Path(args.sample)

    if not reference_path.is_absolute():
        reference_path = PROJECT_ROOT / reference_path
    if not sample_path.is_absolute():
        sample_path = PROJECT_ROOT / sample_path

    reference_rgb = read_image(reference_path)
    sample_rgb = read_image(sample_path)

    reference_lab = robust_average_lab(reference_rgb)
    sample_lab = robust_average_lab(sample_rgb)

    delta_e = float(
        deltaE_ciede2000(
            reference_lab.reshape(1, 1, 3),
            sample_lab.reshape(1, 1, 3),
        )[0, 0]
    )

    status = "WARNING" if delta_e > args.threshold else "PASS"

    print("=" * 50)
    print(f"Reference Lab : {reference_lab.round(3)}")
    print(f"Sample Lab    : {sample_lab.round(3)}")
    print(f"Delta E 2000  : {delta_e:.3f}")
    print(f"Threshold     : {args.threshold:.3f}")
    print(f"Result        : {status}")
    print("=" * 50)


if __name__ == "__main__":
    main()

#测试时准备两张相同拍摄条件下的图片：
# demo_inputs/images/reference.jpg
# demo_inputs/images/sample.jpg

# run:
# python scripts/color_difference_demo.py --reference demo_inputs/images/reference.jpg --sample demo_inputs/images/sample.jpg