# Industrial AI Vision based on YOLO11

> An Industrial AI Vision System for **Aluminium Profile Inspection**, **Inventory Counting**, and **Color Difference Detection** based on **YOLO11** and **OpenCV**.

---

## 📌 Project Introduction

This project aims to build an intelligent industrial vision system for aluminium profile manufacturing and warehouse management.

Current development focuses on two application scenarios:

- 📦 Aluminium Profile Inventory Counting
- 🎨 Aluminium Surface Color Difference Detection

The system is developed based on **YOLO11**, **OpenCV**, and **Python**, and is designed to support future deployment on industrial cameras and edge devices.

---

# ✨ Features

## ✅ Current Features

- YOLO11 environment setup
- Pretrained object detection demo
- Batch image inference
- Inventory counting framework
- Excel report export
- Color difference calculation (Lab + ΔE2000)
- Dataset split utility

---

## 🚧 Planned Features

- Aluminium profile classification
- Surface defect detection
- Automatic inventory counting
- ROI-based color comparison
- Industrial camera integration
- Real-time detection
- Streamlit/PyQt visualization
- Edge deployment (TensorRT / ONNX)

---

# 🏗 Project Workflow

## 1. Inventory Counting

```text
Image
   │
   ▼
YOLO11 Detection
   │
   ▼
Category Recognition
   │
   ▼
Object Counting
   │
   ▼
Excel Report
```

---

## 2. Color Difference Detection

```text
Reference Image
          │
          ▼
     YOLO Detection
          │
          ▼
      ROI Extraction
          │
          ▼
 Remove Background
          │
          ▼
      RGB → Lab
          │
          ▼
     ΔE2000 Calculation
          │
          ▼
 PASS / WARNING
```

---

## 3. Future Industrial Workflow

```text
Industrial Camera
        │
        ▼
YOLO11 Detection
        │
        ▼
Product Classification
        │
        ├─────────────┐
        ▼             ▼
Inventory Count   Surface Inspection
        │             │
        ▼             ▼
 Excel Report   Color Warning / Defect Alarm
```

---

# 📂 Project Structure

```text
Industrial_AI_Vision/
│
├── configs/                  # Dataset configuration
│   └── quality_data.yaml
│
├── dataset/                  # Training dataset (not uploaded)
│   ├── images/
│   └── labels/
│
├── demo_inputs/              # Demo images/videos
│   ├── images/
│   └── videos/
│
├── demo_outputs/             # Detection results
│
├── runs/                     # YOLO training outputs
│
├── scripts/
│   ├── check_environment.py
│   ├── demo_pretrained.py
│   ├── train.py
│   ├── validate.py
│   ├── inventory_count.py
│   ├── color_difference_demo.py
│   └── split_dataset.py
│
├── utils/                    # Utility functions
│
├── README.md
├── requirements.txt
└── .gitignore
```

---

# ⚙ Environment Setup

## 1. Clone Repository

```bash
git clone https://github.com/yourname/Industrial_AI_Vision.git

cd Industrial_AI_Vision
```

---

## 2. Create Conda Environment

```bash
conda create -n industrial_yolo11 python=3.10

conda activate industrial_yolo11
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

or

```bash
pip install ultralytics opencv-python pandas numpy scikit-image openpyxl tqdm
```

---

# 🚀 Quick Start

## Check Environment

```bash
python scripts/check_environment.py
```

Check whether:

- Python
- OpenCV
- Ultralytics
- PyTorch
- CUDA

are correctly installed.

---

## YOLO11 Pretrained Detection

Single image:

```bash
python scripts/demo_pretrained.py --source demo_inputs/images/general_test.jpg
```

Batch images:

```bash
python scripts/demo_pretrained.py --source demo_inputs/images
```

---

## Inventory Counting

```bash
python scripts/inventory_count.py \
--weights yolo11n.pt \
--source demo_inputs/images/general_test.jpg
```

Output:

- Detection image
- Inventory statistics
- Excel report

---

## Color Difference Detection

```bash
python scripts/color_difference_demo.py \
--reference demo_inputs/images/reference.jpg \
--sample demo_inputs/images/sample.jpg
```

Example Output

```text
Reference Lab : [73.508 -1.273  2.380]

Sample Lab    : [46.699  1.283  7.559]

Delta E 2000  : 24.203

Result         : WARNING
```

---

## Dataset Split

```bash
python scripts/split_dataset.py
```

Automatically generate

- train
- validation
- test

datasets.

---

## Model Training

```bash
python scripts/train.py \
--data configs/quality_data.yaml
```

Training results will be saved in

```text
runs/train/
```

---

## Model Validation

```bash
python scripts/validate.py \
--weights runs/train/weights/best.pt \
--data configs/quality_data.yaml
```

Output metrics:

- Precision
- Recall
- mAP50
- mAP50-95

---

# 📊 Current Development Status

| Module | Status |
|---------|--------|
| Environment Setup | ✅ |
| YOLO11 Installation | ✅ |
| Project Framework | ✅ |
| Environment Check | ✅ |
| Pretrained Detection | ✅ |
| Batch Image Detection | ✅ |
| Inventory Counting Framework | ✅ |
| Excel Export | ✅ |
| Color Difference Demo | ✅ |
| Dataset Split Tool | ✅ |
| Custom Model Training | ⏳ |
| Aluminium Profile Recognition | ⏳ |
| Surface Defect Detection | ⏳ |
| Industrial Camera Integration | ⏳ |
| Real-time Deployment | ⏳ |

---

# 🛣 Roadmap

## Phase 1

- [x] Project framework
- [x] Environment setup
- [x] YOLO11 demo
- [x] Inventory counting framework
- [x] Color difference demo

---

## Phase 2

- [ ] Customer dataset collection
- [ ] Data annotation
- [ ] YOLO11 training
- [ ] Model evaluation

---

## Phase 3

- [ ] Aluminium profile classification
- [ ] Surface defect detection
- [ ] Inventory statistics
- [ ] Color warning

---

## Phase 4

- [ ] Industrial camera
- [ ] Streamlit interface
- [ ] Edge deployment

---

# 🛠 Tech Stack

- Python 3.10
- YOLO11 (Ultralytics)
- OpenCV
- NumPy
- Pandas
- Scikit-image
- PyTorch

---

# 📄 License

This project is released under the MIT License.

---

# 👨‍💻 Author

Industrial AI Vision Project

YOLO11 + OpenCV + Python