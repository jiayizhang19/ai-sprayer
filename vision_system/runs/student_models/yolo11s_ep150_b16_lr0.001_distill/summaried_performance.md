# Knowledge Distillation Performance Summary
**Date:** 2026-08-12 17:32  
**Teacher:** `best.pt`  
**Student baseline:** `best.pt`  
**Student distilled:** `yolo11s_ep150_b16_lr0.001_distill`  

---

## 1. Model Size Comparison

| Model | Parameters (M) | File Size (MB) | Notes |
|:------|:--------------:|:--------------:|:------|
| **yolo11m (Teacher)** | 20.09 | 38.7 | Upper-bound accuracy |
| **yolo11s (Baseline)** | 9.45 | 18.4 | Same architecture, no KD |
| **yolo11s (Distilled)** | 9.45 | 18.4 | Same architecture + KD |

> Distilled and baseline yolo11s should have **identical** parameter count and nearly identical file size.

---

## 2. Overall Accuracy (Validation Split)

| Metric | yolo11m<br>(Teacher) | yolo11s<br>(Baseline) | yolo11s<br>(Distilled) | Δ vs Baseline | Δ vs Teacher |
|:-------|:--------------------:|:---------------------:|:----------------------:|:-------------:|:------------:|
| **Precision** | 0.9077 | 0.9023 | **0.8968** | -0.0056 | -0.0110 |
| **Recall**    | 0.7846    | 0.7829    | **0.7826**    | -0.0003       | -0.0020 |
| **mAP50**     | 0.7946     | 0.7947     | **0.7924**     | -0.0023         | -0.0022 |
| **mAP50-95**  | 0.4849       | 0.4869       | **0.4997**       | +0.0128             | +0.0148 |

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
| Teacher (yolo11m) | `C:\Users\25252980\projects\ai-sprayer\vision_system\runs\train\yolo11m_ep150_b8_lr0.001\weights\best.pt` |
| Baseline (yolo11s) | `C:\Users\25252980\projects\ai-sprayer\vision_system\runs\train\yolo11s_ep150_b8_lr0.001\weights\best.pt` |
| Distilled (yolo11s) | `C:\Users\25252980\projects\ai-sprayer\vision_system\runs\student_models\yolo11s_ep150_b16_lr0.001_distill\weights\best.pt` |

## Notes
- Teacher is frozen during training; only the student + projector are updated.
- Final `best.pt` contains **only the student** → identical parameter count and inference cost to a normal yolo11s.
- Distillation loss weight used: `dis=6.0` (increase to 8–10 for stronger teacher influence).
