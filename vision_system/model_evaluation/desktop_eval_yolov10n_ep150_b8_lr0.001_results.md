# Evaluation Report: YOLO (yolov10n_ep150_b8_lr0.001)

## Core Metrics Summary
- **Precision:** 0.8243
- **Recall:** 0.8330
- **F1 Score:** 0.8286
- **Mean IoU (Matched):** 0.8364

## Bounding Box Breakdown
- **Total Ground Truth Boxes:** 2162
- **Total Predicted Boxes:** 2185
- **True Positives (TP):** 1801
- **False Positives (FP):** 384
- **False Negatives (FN):** 361

## Per-Class Metrics Table
| Class Code | TP | FP | FN | Precision | Recall | F1 Score |
|--- |--- |--- |--- |--- |--- |--- |
| BROST | 232 | 50 | 65 | 0.8227 | 0.7811 | 0.8014 |
| STEME | 768 | 235 | 183 | 0.7657 | 0.8076 | 0.7861 |
| URTUR | 801 | 99 | 113 | 0.8900 | 0.8764 | 0.8831 |

## Inference Timing — Overall
- **Mean:** 49.2 ms  (20.31 FPS)
- **Median:** 19.1 ms
- **Std Dev:** 68.2 ms
- **Mean (excl. first image / warm-up):** 43.4 ms (23.04 FPS)

## Inference Timing — By Class Present in Image
_Note: this groups images by which ground-truth classes they contain; it is the mean total inference time for images containing that class, not a per-detection or per-class model cost (YOLO predicts all classes in a single forward pass per image)._

| Class Code | Mean Image Time (ms) | Images (n) |
|--- |--- |--- |
| BROST | 55.6 | 39 |
| STEME | 46.6 | 37 |
| URTUR | 43.5 | 31 |

## Setup Configuration Context
- **Model Identifier:** `yolov10n_ep150_b8_lr0.001`
- **Eval Platform:** `desktop`
- **Device Used:** `0`
- **Data Type:** `float32`
