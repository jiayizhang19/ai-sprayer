# Evaluation Report: YOLO (yolo11n_ep150_b8_lr0.001)

## Core Metrics Summary
- **Precision:** 0.0000
- **Recall:** 0.0000
- **F1 Score:** 0.0000
- **Mean IoU (Matched):** 0.0000

## Bounding Box Breakdown
- **Total Ground Truth Boxes:** 0
- **Total Predicted Boxes:** 2226
- **True Positives (TP):** 0
- **False Positives (FP):** 2226
- **False Negatives (FN):** 0

## Per-Class Metrics Table
| Class Code | TP | FP | FN | Precision | Recall | F1 Score |
|--- |--- |--- |--- |--- |--- |--- |
| STEME | 0 | 1027 | 0 | 0.0000 | 0.0000 | 0.0000 |
| URTUR | 0 | 912 | 0 | 0.0000 | 0.0000 | 0.0000 |
| BROST | 0 | 287 | 0 | 0.0000 | 0.0000 | 0.0000 |

## Inference Timing — Overall
- **Mean:** 600.9 ms  (1.66 FPS)
- **Median:** 577.1 ms
- **Std Dev:** 193.2 ms
- **Mean (excl. first image / warm-up):** 582.3 ms (1.72 FPS)

## Inference Timing — By Class Present in Image
_Note: this groups images by which ground-truth classes they contain; it is the mean total inference time for images containing that class, not a per-detection or per-class model cost (YOLO predicts all classes in a single forward pass per image)._

| Class Code | Mean Image Time (ms) | Images (n) |
|--- |--- |--- |

## Setup Configuration Context
- **Model Identifier:** `yolo11n_ep150_b8_lr0.001`
- **Eval Platform:** `rpi5`
- **Device Used:** `cpu`
- **Data Type:** `float32`
