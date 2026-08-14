# Knowledge Distillation Performance Summary
**Date:** 2026-08-12 16:03  
**Teacher:** `best.pt`  
**Student baseline:** `best.pt`  
**Student distilled:** `yolov8s_ep150_b16_lr0.001_distill`  

---

## 1. Model Size Comparison

| Model | Parameters (M) | File Size (MB) | Notes |
|:------|:--------------:|:--------------:|:------|
| **YOLOv8m (Teacher)** | 25.88 | 49.7 | Upper-bound accuracy |
| **YOLOv8s (Baseline)** | 11.15 | 21.6 | Same architecture, no KD |
| **YOLOv8s (Distilled)** | 11.15 | 21.6 | Same architecture + KD |

> Distilled and baseline YOLOv8s should have **identical** parameter count and nearly identical file size.

---

## 2. Overall Accuracy (Validation Split)

| Metric | YOLOv8m<br>(Teacher) | YOLOv8s<br>(Baseline) | YOLOv8s<br>(Distilled) | Δ vs Baseline | Δ vs Teacher |
|:-------|:--------------------:|:---------------------:|:----------------------:|:-------------:|:------------:|
| **Precision** | 0.9178 | 0.9107 | **0.9246** | +0.0139 | +0.0068 |
| **Recall**    | 0.8035    | 0.8006    | **0.7868**    | -0.0138       | -0.0168 |
| **mAP50**     | 0.8172     | 0.8088     | **0.7968**     | -0.0119         | -0.0203 |
| **mAP50-95**  | 0.5044       | 0.4980       | **0.4982**       | +0.0002             | -0.0062 |

> Positive Δ means the distilled student improved over that reference.

---

## 3. Class-Specific Performance (Distilled Student)

| Class ID | Class Name | Precision | Recall | mAP50 | mAP50-95 |
|:--------:|:-----------|:---------:|:------:|:-----:|:--------:|
| — | (class metrics not available) | — | — | — | — |

---

## 4. Model Paths

| Role | Path |
|:-----|:-----|
| Teacher (YOLOv8m) | `C:\Users\25252980\Projects\ai-sprayer\vision_system\runs\train\yolov8m_ep150_b8_lr0.001\weights\best.pt` |
| Baseline (YOLOv8s) | `C:\Users\25252980\Projects\ai-sprayer\vision_system\runs\train\yolov8s_ep150_b8_lr0.001\weights\best.pt` |
| Distilled (YOLOv8s) | `C:\Users\25252980\Projects\ai-sprayer\vision_system\runs\student_models\yolov8s_ep150_b16_lr0.001_distill\weights\best.pt` |

## Notes
- Teacher is frozen during training; only the student + projector are updated.
- Final `best.pt` contains **only the student** → identical parameter count and inference cost to a normal YOLOv8s.
- Distillation loss weight used: `dis=6.0` (increase to 8–10 for stronger teacher influence).
