# Evaluation Report: YOLO (yolov8n_ep150_b8_lr0.001)

## Core Metrics Summary
- **Precision:** 0.8245
- **Recall:** 0.8608
- **F1 Score:** 0.8423
- **Mean IoU (Matched):** 0.8339

## Bounding Box Breakdown
- **Total Ground Truth Boxes:** 2162
- **Total Predicted Boxes:** 2257
- **True Positives (TP):** 1861
- **False Positives (FP):** 396
- **False Negatives (FN):** 301

## Per-Class Metrics Table
| Class Code | TP | FP | FN | Precision | Recall | F1 Score |
|--- |--- |--- |--- |--- |--- |--- |
| BROST | 239 | 66 | 58 | 0.7836 | 0.8047 | 0.7940 |
| STEME | 790 | 259 | 161 | 0.7531 | 0.8307 | 0.7900 |
| URTUR | 832 | 71 | 82 | 0.9214 | 0.9103 | 0.9158 |

## Inference Timing — Overall
### End-to-End (wall-clock, includes pre/post-process + Python overhead)
- **Mean:** 42.8 ms  (23.35 FPS)
- **Median:** 12.0 ms
- **Std Dev:** 58.3 ms
- **Mean (excl. first image / warm-up):** 38.2 ms (26.18 FPS)

### Pure Neural-Network Inference (Ultralytics `result.speed['inference']`)
- **Mean:** 6.4 ms  (156.90 FPS)
- **Median:** 4.2 ms
- **Std Dev:** 6.7 ms
- **Mean (excl. first image / warm-up):** 6.4 ms (156.56 FPS)

## Inference Timing — By Class Present in Image
_Note: this groups images by which ground-truth classes they contain; it is the mean total inference time for images containing that class, not a per-detection or per-class model cost (YOLO predicts all classes in a single forward pass per image)._

| Class Code | Mean Image Time (ms) | Images (n) |
|--- |--- |--- |
| BROST | 51.2 | 39 |
| STEME | 38.5 | 37 |
| URTUR | 36.7 | 31 |

## Setup Configuration Context
- **Model Identifier:** `yolov8n_ep150_b8_lr0.001`
- **Eval Platform:** `desktop`
- **Device Used:** `0`
- **Data Type:** `float32`
