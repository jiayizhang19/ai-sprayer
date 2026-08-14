# Knowledge Distillation Performance Summary
**Date:** 2026-08-12 23:56  
**Teacher:** `best.pt`  
**Student baseline:** `best.pt`  
**Student distilled:** `yolo26s_ep150_b16_lr0.001_distill`  

---

## 1. Model Size Comparison

| Model | Parameters (M) | File Size (MB) | Notes |
|:------|:--------------:|:--------------:|:------|
| **YOLO26m (Teacher)** | 21.84 | 42.2 | Upper-bound accuracy |
| **YOLO26s (Baseline)** | 9.98 | 19.5 | Same architecture, no KD |
| **YOLO26s (Distilled)** | 9.98 | 19.5 | Same architecture + KD |

> Distilled and baseline YOLO26s should have **identical** parameter count and nearly identical file size.

---

## 2. Overall Accuracy (Validation Split)

| Metric | YOLO26m<br>(Teacher) | YOLO26s<br>(Baseline) | YOLO26s<br>(Distilled) | Δ vs Baseline | Δ vs Teacher |
|:-------|:--------------------:|:---------------------:|:----------------------:|:-------------:|:------------:|
| **Precision** | 0.9041 | 0.9050 | **0.9075** | +0.0025 | +0.0034 |
| **Recall**    | 0.7831    | 0.7899    | **0.7719**    | -0.0180       | -0.0112 |
| **mAP50**     | 0.8047     | 0.7985     | **0.7920**     | -0.0065         | -0.0127 |
| **mAP50-95**  | 0.4943       | 0.5046       | **0.4818**       | -0.0228             | -0.0125 |

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
| Teacher (YOLO26m) | `C:\Users\25252980\Projects\ai-sprayer\vision_system\runs\train\yolo26m_ep150_b8_lr0.001\weights\best.pt` |
| Baseline (YOLO26s) | `C:\Users\25252980\Projects\ai-sprayer\vision_system\runs\train\yolo26s_ep150_b8_lr0.001\weights\best.pt` |
| Distilled (YOLO26s) | `C:\Users\25252980\Projects\ai-sprayer\vision_system\runs\student_models\yolo26s_ep150_b16_lr0.001_distill\weights\best.pt` |

## Notes
- Teacher is frozen during training; only the student + projector are updated.
- Final `best.pt` contains **only the student** → identical parameter count and inference cost to a normal YOLO26s.
- Distillation loss weight used: `dis=6.0` (increase to 8–10 for stronger teacher influence).
