# Evaluation Report: YOLO (yolov8s_ep150_b8_lr0.001_fp16)

## Core Metrics Summary
- **Precision:** 0.8329
- **Recall:** 0.8552
- **F1 Score:** 0.8439
- **Mean IoU (Matched):** 0.8382

## Bounding Box Breakdown
- **Total Ground Truth Boxes:** 2162
- **Total Predicted Boxes:** 2220
- **True Positives (TP):** 1849
- **False Positives (FP):** 371
- **False Negatives (FN):** 313

## Per-Class Metrics Table
| Class Code | TP | FP | FN | Precision | Recall | F1 Score |
|--- |--- |--- |--- |--- |--- |--- |
| URTUR | 827 | 133 | 87 | 0.8615 | 0.9048 | 0.8826 |
| STEME | 779 | 211 | 172 | 0.7869 | 0.8191 | 0.8027 |
| BROST | 243 | 27 | 54 | 0.9000 | 0.8182 | 0.8571 |

## Inference Timing — Overall
### End-to-End (wall-clock, includes pre/post-process + Python overhead)
- **Mean:** 94.1 ms  (10.62 FPS)
- **Median:** 33.8 ms
- **Std Dev:** 209.9 ms
- **Mean (excl. first image / warm-up):** 74.6 ms (13.40 FPS)

### Pure Neural-Network Inference (Ultralytics `result.speed['inference']`)
- **Mean:** 12.2 ms  (81.87 FPS)
- **Median:** 12.0 ms
- **Std Dev:** 4.3 ms
- **Mean (excl. first image / warm-up):** 12.2 ms (81.86 FPS)

## Inference Timing — By Class Present in Image
_Note: this groups images by which ground-truth classes they contain; it is the mean total inference time for images containing that class, not a per-detection or per-class model cost (YOLO predicts all classes in a single forward pass per image)._

| Class Code | Mean Image Time (ms) | Images (n) |
|--- |--- |--- |
| STEME | 133.9 | 37 |
| URTUR | 79.2 | 31 |
| BROST | 67.4 | 39 |

## Setup Configuration Context
- **Model Identifier:** `yolov8s_ep150_b8_lr0.001_fp16`
- **Eval Platform:** `Orin-engine`
- **Device Used:** `0`
- **Data Type:** `float32`
