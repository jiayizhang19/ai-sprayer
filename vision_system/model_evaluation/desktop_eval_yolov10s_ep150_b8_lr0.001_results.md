# Evaluation Report: YOLO (yolov10s_ep150_b8_lr0.001)

## Core Metrics Summary
- **Precision:** 0.7897
- **Recall:** 0.8492
- **F1 Score:** 0.8184
- **Mean IoU (Matched):** 0.8229

## Bounding Box Breakdown
- **Total Ground Truth Boxes:** 2162
- **Total Predicted Boxes:** 2325
- **True Positives (TP):** 1836
- **False Positives (FP):** 489
- **False Negatives (FN):** 326

## Per-Class Metrics Table
| Class Code | TP | FP | FN | Precision | Recall | F1 Score |
|--- |--- |--- |--- |--- |--- |--- |
| BROST | 244 | 82 | 53 | 0.7485 | 0.8215 | 0.7833 |
| STEME | 763 | 275 | 188 | 0.7351 | 0.8023 | 0.7672 |
| URTUR | 829 | 132 | 85 | 0.8626 | 0.9070 | 0.8843 |

## Inference Timing — Overall
- **Mean:** 66.2 ms  (15.11 FPS)
- **Median:** 29.6 ms
- **Std Dev:** 84.9 ms
- **Mean (excl. first image / warm-up):** 58.8 ms (17.01 FPS)

## Inference Timing — By Class Present in Image
_Note: this groups images by which ground-truth classes they contain; it is the mean total inference time for images containing that class, not a per-detection or per-class model cost (YOLO predicts all classes in a single forward pass per image)._

| Class Code | Mean Image Time (ms) | Images (n) |
|--- |--- |--- |
| BROST | 74.4 | 39 |
| STEME | 62.1 | 37 |
| URTUR | 59.7 | 31 |

## Setup Configuration Context
- **Model Identifier:** `yolov10s_ep150_b8_lr0.001`
- **Eval Platform:** `desktop`
- **Device Used:** `0`
- **Data Type:** `float32`
