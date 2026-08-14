# Evaluation Report: YOLO (yolo26s_ep150_b16_lr0.001_distill)

## Core Metrics Summary
- **Precision:** 0.8573
- **Recall:** 0.8640
- **F1 Score:** 0.8606
- **Mean IoU (Matched):** 0.8250

## Bounding Box Breakdown
- **Total Ground Truth Boxes:** 2162
- **Total Predicted Boxes:** 2179
- **True Positives (TP):** 1868
- **False Positives (FP):** 311
- **False Negatives (FN):** 294

## Per-Class Metrics Table
| Class Code | TP | FP | FN | Precision | Recall | F1 Score |
|--- |--- |--- |--- |--- |--- |--- |
| BROST | 241 | 44 | 56 | 0.8456 | 0.8114 | 0.8282 |
| STEME | 786 | 173 | 165 | 0.8196 | 0.8265 | 0.8230 |
| URTUR | 841 | 94 | 73 | 0.8995 | 0.9201 | 0.9097 |

## Inference Timing — Overall
### End-to-End (wall-clock, includes pre/post-process + Python overhead)
- **Mean:** 66.9 ms  (14.95 FPS)
- **Median:** 16.2 ms
- **Std Dev:** 103.3 ms
- **Mean (excl. first image / warm-up):** 58.6 ms (17.06 FPS)

### Pure Neural-Network Inference (Ultralytics `result.speed['inference']`)
- **Mean:** 21.3 ms  (47.05 FPS)
- **Median:** 8.2 ms
- **Std Dev:** 28.0 ms
- **Mean (excl. first image / warm-up):** 21.4 ms (46.77 FPS)

## Inference Timing — By Class Present in Image
_Note: this groups images by which ground-truth classes they contain; it is the mean total inference time for images containing that class, not a per-detection or per-class model cost (YOLO predicts all classes in a single forward pass per image)._

| Class Code | Mean Image Time (ms) | Images (n) |
|--- |--- |--- |
| BROST | 96.2 | 39 |
| STEME | 49.2 | 37 |
| URTUR | 49.2 | 31 |

## Setup Configuration Context
- **Model Identifier:** `yolo26s_ep150_b16_lr0.001_distill`
- **Eval Platform:** `desktop`
- **Device Used:** `0`
- **Data Type:** `float32`
