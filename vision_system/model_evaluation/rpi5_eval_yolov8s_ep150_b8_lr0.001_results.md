# Evaluation Report: YOLO (yolov8s_ep150_b8_lr0.001)

## Core Metrics Summary
- **Precision:** 0.0000
- **Recall:** 0.0000
- **F1 Score:** 0.0000
- **Mean IoU (Matched):** 0.0000

## Bounding Box Breakdown
- **Total Ground Truth Boxes:** 0
- **Total Predicted Boxes:** 2188
- **True Positives (TP):** 0
- **False Positives (FP):** 2188
- **False Negatives (FN):** 0

## Per-Class Metrics Table
| Class Code | TP | FP | FN | Precision | Recall | F1 Score |
|--- |--- |--- |--- |--- |--- |--- |
| URTUR | 0 | 942 | 0 | 0.0000 | 0.0000 | 0.0000 |
| STEME | 0 | 959 | 0 | 0.0000 | 0.0000 | 0.0000 |
| BROST | 0 | 287 | 0 | 0.0000 | 0.0000 | 0.0000 |

## Inference Timing — Overall
- **Mean:** 1318.8 ms  (0.76 FPS)
- **Median:** 1304.4 ms
- **Std Dev:** 236.5 ms
- **Mean (excl. first image / warm-up):** 1298.4 ms (0.77 FPS)

## Inference Timing — By Class Present in Image
_Note: this groups images by which ground-truth classes they contain; it is the mean total inference time for images containing that class, not a per-detection or per-class model cost (YOLO predicts all classes in a single forward pass per image)._

| Class Code | Mean Image Time (ms) | Images (n) |
|--- |--- |--- |

## Setup Configuration Context
- **Model Identifier:** `yolov8s_ep150_b8_lr0.001`
- **Eval Platform:** `rpi5`
- **Device Used:** `cpu`
- **Data Type:** `float32`
