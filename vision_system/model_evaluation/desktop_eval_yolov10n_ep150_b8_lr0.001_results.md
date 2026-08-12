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
### End-to-End (wall-clock, includes pre/post-process + Python overhead)
- **Mean:** 38.9 ms  (25.70 FPS)
- **Median:** 12.1 ms
- **Std Dev:** 47.4 ms
- **Mean (excl. first image / warm-up):** 35.3 ms (28.35 FPS)

### Pure Neural-Network Inference (Ultralytics `result.speed['inference']`)
- **Mean:** 6.5 ms  (152.90 FPS)
- **Median:** 5.4 ms
- **Std Dev:** 5.7 ms
- **Mean (excl. first image / warm-up):** 6.5 ms (152.96 FPS)

## Inference Timing — By Class Present in Image
_Note: this groups images by which ground-truth classes they contain; it is the mean total inference time for images containing that class, not a per-detection or per-class model cost (YOLO predicts all classes in a single forward pass per image)._

| Class Code | Mean Image Time (ms) | Images (n) |
|--- |--- |--- |
| BROST | 43.0 | 39 |
| STEME | 37.1 | 37 |
| URTUR | 35.1 | 31 |

## Setup Configuration Context
- **Model Identifier:** `yolov10n_ep150_b8_lr0.001`
- **Eval Platform:** `desktop`
- **Device Used:** `0`
- **Data Type:** `float32`
