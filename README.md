# Scripts Overview

本项目基于 **YOLO11 + OpenCV** 搭建工业 AI 视觉检测系统，目前包含环境检查、目标检测、模型训练、货架盘点、色差检测等多个模块。

---

## 1. Environment Check

### Script

```bash
python scripts/check_environment.py
```

### Function

检查当前开发环境是否配置正确，包括：

- Python 版本
- Ultralytics 版本
- OpenCV 版本
- PyTorch 版本
- CUDA 是否可用
- GPU 信息

运行成功表示：

- Python 环境正常
- YOLO11 可以正常调用
- 后续可以进行模型训练与推理

---

## 2. YOLO11 Pretrained Demo

### Script

```bash
python scripts/demo_pretrained.py --source demo_inputs/images/general_test.jpg --device cpu
```

或

```bash
python scripts/demo_pretrained.py --source demo_inputs/images
```

### Function

使用 **YOLO11 预训练模型**进行目标检测。

功能包括：

- 加载 YOLO11 预训练模型
- 图片读取
- 自动目标检测
- 绘制检测框
- 保存检测结果

输出目录：

```

demo\_outputs/pretrained\_detection/

```

说明：

目前使用 COCO 预训练模型，仅能识别通用类别（person、car、truck 等），不能识别工业铝型材。

---

## 3. Model Training

### Script

```bash
python scripts/train.py --data configs/quality_data.yaml
```

### Function

训练自定义 YOLO11 模型。

主要流程：

- 加载预训练权重
- 读取数据集
- 开始训练
- 自动保存 best.pt

训练完成后模型保存在：

```

runs/train/

```

目前由于尚未提供数据集，此功能尚未进行实际训练测试。

---

## 4. Model Validation

### Script

```bash
python scripts/validate.py --weights runs/train/weights/best.pt --data configs/quality_data.yaml
```

### Function

验证训练后的模型性能。

输出指标包括：

- Precision
- Recall
- mAP50
- mAP50-95

目前等待模型训练完成后进行测试。

---

## 5. Inventory Counting

### Script

```bash
python scripts/inventory_count.py --weights yolo11n.pt --source demo_inputs/images/general_test.jpg --device cpu
```

### Function

实现货架产品数量统计。

流程：

```

Image
↓
YOLO Detection
↓
Category Extraction
↓
Count Objects
↓
Export Excel

```

输出：

- 检测结果图片
- 产品数量统计
- Excel 报表

目前由于未训练工业数据，仅能统计 COCO 类别。

后续训练完成后可统计：

- 铝型材A
- 铝型材B
- 铝型材C
- ...

---

## 6. Color Difference Demo

### Script

```bash
python scripts/color_difference_demo.py --reference demo_inputs/images/reference.jpg --sample demo_inputs/images/sample.jpg
```

### Function

实现产品颜色比较。

流程：

```

Reference Image
↓
RGB → Lab
↓
ΔE2000 Calculation
↓
PASS / WARNING

```

测试结果：

### Test 1

Reference VS Reference

```

DeltaE = 0.000
PASS

```

### Test 2

Reference VS Sample

```

DeltaE = 24.203
WARNING

```

说明：

目前为 Demo，仅计算整张图片代表颜色。

后续将升级为：

```

YOLO Detection
↓
ROI Extraction
↓
Remove Background
↓
Remove Reflection
↓
Lab
↓
ΔE2000
↓
Color Warning

```

---

## 7. Dataset Split

### Script

```bash
python scripts/split_dataset.py
```

### Function

自动划分数据集。

功能：

- Train
- Validation
- Test

默认比例：

```

8 : 1 : 1

```

适用于甲方数据标注完成后的数据整理。

---

# Current Development Status

| Module | Status |
|---------|--------|
| Development Environment | ✅ Completed |
| YOLO11 Installation | ✅ Completed |
| Project Structure | ✅ Completed |
| Environment Check | ✅ Completed |
| Pretrained Detection Demo | ✅ Completed |
| Image Detection | ✅ Completed |
| Batch Image Detection | ✅ Completed |
| Inventory Counting Framework | ✅ Completed |
| Excel Export | ✅ Completed |
| Color Difference Demo | ✅ Completed |
| Dataset Split Tool | ✅ Completed |
| Industrial Dataset Training | ⏳ Waiting for dataset |
| Aluminium Profile Recognition | ⏳ Waiting for dataset |
| Surface Defect Detection | ⏳ Waiting for dataset |
| Real Shelf Inventory | ⏳ Waiting for dataset |
| Industrial Camera Integration | ⏳ Future Work |

---

# Next Step

After receiving the customer's dataset:

1. Data annotation (LabelImg / CVAT)
2. Convert to YOLO format
3. Dataset split
4. Train YOLO11
5. Model evaluation
6. Aluminium profile classification
7. Surface defect detection
8. Automatic inventory counting
9. Color difference warning
10. Industrial deployment
