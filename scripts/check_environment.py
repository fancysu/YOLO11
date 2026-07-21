from __future__ import annotations

import platform
import sys

import cv2
import torch
import ultralytics


def main() -> None:
    """检查工业视觉项目运行环境。"""

    print("=" * 60)
    print("Industrial AI Vision Environment Check")
    print("=" * 60)

    print(f"Operating system : {platform.platform()}")
    print(f"Python           : {sys.version.split()[0]}")
    print(f"Ultralytics      : {ultralytics.__version__}")
    print(f"OpenCV           : {cv2.__version__}")
    print(f"PyTorch          : {torch.__version__}")
    print(f"CUDA available   : {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"CUDA version     : {torch.version.cuda}")
        print(f"GPU count        : {torch.cuda.device_count()}")
        print(f"GPU name         : {torch.cuda.get_device_name(0)}")
        print("Recommended device: 0")
    else:
        print("GPU not detected. The demo can use CPU, but training will be slower.")
        print("Recommended device: cpu")

    print("=" * 60)
    print("Environment check completed.")


if __name__ == "__main__":
    main()