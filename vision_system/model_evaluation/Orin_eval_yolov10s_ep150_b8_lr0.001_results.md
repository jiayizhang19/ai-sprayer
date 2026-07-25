# Evaluation Report: YOLO (yolov10s_ep150_b8_lr0.001)

## Core Metrics Summary
- **Precision:** 0.7897
- **Recall:** 0.8492
- **F1 Score:** 0.8184
- **Mean IoU (Matched):** 0.8230

## Bounding Box Breakdown
- **Total Ground Truth Boxes:** 2162
- **Total Predicted Boxes:** 2325
- **True Positives (TP):** 1836
- **False Positives (FP):** 489
- **False Negatives (FN):** 326

## Per-Class Metrics Table
| Class Code | TP | FP | FN | Precision | Recall | F1 Score |
|--- |--- |--- |--- |--- |--- |--- |
| STEME | 763 | 274 | 188 | 0.7358 | 0.8023 | 0.7676 |
| URTUR | 829 | 132 | 85 | 0.8626 | 0.9070 | 0.8843 |
| BROST | 244 | 83 | 53 | 0.7462 | 0.8215 | 0.7821 |

## Inference Timing — Overall
- **Mean:** 110.1 ms  (9.09 FPS)
- **Median:** 63.0 ms
- **Std Dev:** 106.6 ms
- **Mean (excl. first image / warm-up):** 101.9 ms (9.81 FPS)

## Inference Timing — By Class Present in Image
_Note: this groups images by which ground-truth classes they contain; it is the mean total inference time for images containing that class, not a per-detection or per-class model cost (YOLO predicts all classes in a single forward pass per image)._

| Class Code | Mean Image Time (ms) | Images (n) |
|--- |--- |--- |
| STEME | 127.6 | 37 |
| URTUR | 101.3 | 31 |
| BROST | 98.9 | 39 |

## Setup Configuration Context
- **Model Identifier:** `yolov10s_ep150_b8_lr0.001`
- **Eval Platform:** `Orin`
- **Device Used:** `0`
- **Data Type:** `float32`
