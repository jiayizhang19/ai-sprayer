# Knowledge Distillation Performance Summary
**Date:** 2026-08-13 02:30  
**Teacher:** `best.pt`  
**Student baseline:** `best.pt`  
**Student distilled:** `yolov10s_ep150_b16_lr0.001_distill`  

---

## 1. Model Size Comparison

| Model | Parameters (M) | File Size (MB) | Notes |
|:------|:--------------:|:--------------:|:------|
| **yolov10m (Teacher)** | 16.54 | 32.1 | Upper-bound accuracy |
| **yolov10s (Baseline)** | 8.10 | 15.9 | Same architecture, no KD |
| **yolov10s (Distilled)** | 8.10 | 15.9 | Same architecture + KD |

> Distilled and baseline yolov10s should have **identical** parameter count and nearly identical file size.

---

## 2. Overall Accuracy (Validation Split)

| Metric | yolov10m<br>(Teacher) | yolov10s<br>(Baseline) | yolov10s<br>(Distilled) | Δ vs Baseline | Δ vs Teacher |
|:-------|:--------------------:|:---------------------:|:----------------------:|:-------------:|:------------:|
| **Precision** | 0.8955 | 0.8964 | **0.8929** | -0.0035 | -0.0027 |
| **Recall**    | 0.7754    | 0.7713    | **0.7649**    | -0.0065       | -0.0106 |
| **mAP50**     | 0.7910     | 0.7920     | **0.7775**     | -0.0145         | -0.0135 |
| **mAP50-95**  | 0.4780       | 0.4845       | **0.4723**       | -0.0122             | -0.0058 |

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
| Teacher (yolov10m) | `C:\Users\25252980\Projects\ai-sprayer\vision_system\runs\train\yolov10m_ep150_b8_lr0.001\weights\best.pt` |
| Baseline (yolov10s) | `C:\Users\25252980\Projects\ai-sprayer\vision_system\runs\train\yolov10s_ep150_b8_lr0.001\weights\best.pt` |
| Distilled (yolov10s) | `C:\Users\25252980\Projects\ai-sprayer\vision_system\runs\student_models\yolov10s_ep150_b16_lr0.001_distill\weights\best.pt` |

## Notes
- Teacher is frozen during training; only the student + projector are updated.
- Final `best.pt` contains **only the student** → identical parameter count and inference cost to a normal yolov10s.
- Distillation loss weight used: `dis=6.0` (increase to 8–10 for stronger teacher influence).
