# Evaluation Report: YOLO (yolov8n_ep150_b8_lr0.001)

## Core Metrics Summary
- **Precision:** 0.8245
- **Recall:** 0.8608
- **F1 Score:** 0.8423
- **Mean IoU (Matched):** 0.8340

## Bounding Box Breakdown
- **Total Ground Truth Boxes:** 2162
- **Total Predicted Boxes:** 2257
- **True Positives (TP):** 1861
- **False Positives (FP):** 396
- **False Negatives (FN):** 301

## Per-Class Metrics Table
| Class Code | TP | FP | FN | Precision | Recall | F1 Score |
|--- |--- |--- |--- |--- |--- |--- |
| STEME | 790 | 259 | 161 | 0.7531 | 0.8307 | 0.7900 |
| URTUR | 832 | 71 | 82 | 0.9214 | 0.9103 | 0.9158 |
| BROST | 239 | 66 | 58 | 0.7836 | 0.8047 | 0.7940 |

## Inference Timing — Overall
### End-to-End (wall-clock, includes pre/post-process + Python overhead)
- **Mean:** 114.1 ms  (8.76 FPS)
- **Median:** 48.0 ms
- **Std Dev:** 245.0 ms
- **Mean (excl. first image / warm-up):** 91.2 ms (10.96 FPS)

### Pure Neural-Network Inference (Ultralytics `result.speed['inference']`)
- **Mean:** 25.7 ms  (38.89 FPS)
- **Median:** 24.5 ms
- **Std Dev:** 11.4 ms
- **Mean (excl. first image / warm-up):** 24.9 ms (40.19 FPS)

## Inference Timing — By Class Present in Image
_Note: this groups images by which ground-truth classes they contain; it is the mean total inference time for images containing that class, not a per-detection or per-class model cost (YOLO predicts all classes in a single forward pass per image)._

| Class Code | Mean Image Time (ms) | Images (n) |
|--- |--- |--- |
| STEME | 160.2 | 37 |
| URTUR | 91.2 | 31 |
| BROST | 87.8 | 39 |

## Setup Configuration Context
- **Model Identifier:** `yolov8n_ep150_b8_lr0.001`
- **Eval Platform:** `Orin`
- **Device Used:** `0`
- **Data Type:** `float32`
