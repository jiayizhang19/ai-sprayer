# Evaluation Report: YOLO (yolo26n_ep150_b8_lr0.001)

## Core Metrics Summary
- **Precision:** 0.8544
- **Recall:** 0.8252
- **F1 Score:** 0.8395
- **Mean IoU (Matched):** 0.8404

## Bounding Box Breakdown
- **Total Ground Truth Boxes:** 2162
- **Total Predicted Boxes:** 2088
- **True Positives (TP):** 1784
- **False Positives (FP):** 304
- **False Negatives (FN):** 378

## Per-Class Metrics Table
| Class Code | TP | FP | FN | Precision | Recall | F1 Score |
|--- |--- |--- |--- |--- |--- |--- |
| BROST | 233 | 37 | 64 | 0.8630 | 0.7845 | 0.8219 |
| STEME | 743 | 181 | 208 | 0.8041 | 0.7813 | 0.7925 |
| URTUR | 808 | 86 | 106 | 0.9038 | 0.8840 | 0.8938 |

## Inference Timing — Overall
- **Mean:** 56.7 ms  (17.62 FPS)
- **Median:** 24.8 ms
- **Std Dev:** 74.9 ms
- **Mean (excl. first image / warm-up):** 50.4 ms (19.86 FPS)

## Inference Timing — By Class Present in Image
_Note: this groups images by which ground-truth classes they contain; it is the mean total inference time for images containing that class, not a per-detection or per-class model cost (YOLO predicts all classes in a single forward pass per image)._

| Class Code | Mean Image Time (ms) | Images (n) |
|--- |--- |--- |
| BROST | 62.5 | 39 |
| STEME | 54.0 | 37 |
| URTUR | 52.2 | 31 |

## Setup Configuration Context
- **Model Identifier:** `yolo26n_ep150_b8_lr0.001`
- **Eval Platform:** `desktop`
- **Device Used:** `0`
- **Data Type:** `float32`
