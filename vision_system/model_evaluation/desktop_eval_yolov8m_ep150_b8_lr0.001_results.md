# Evaluation Report: YOLO (yolov8m_ep150_b8_lr0.001)

## Core Metrics Summary
- **Precision:** 0.8646
- **Recall:** 0.8834
- **F1 Score:** 0.8739
- **Mean IoU (Matched):** 0.8313

## Bounding Box Breakdown
- **Total Ground Truth Boxes:** 2162
- **Total Predicted Boxes:** 2209
- **True Positives (TP):** 1910
- **False Positives (FP):** 299
- **False Negatives (FN):** 252

## Per-Class Metrics Table
| Class Code | TP | FP | FN | Precision | Recall | F1 Score |
|--- |--- |--- |--- |--- |--- |--- |
| BROST | 250 | 59 | 47 | 0.8091 | 0.8418 | 0.8251 |
| STEME | 806 | 150 | 145 | 0.8431 | 0.8475 | 0.8453 |
| URTUR | 854 | 90 | 60 | 0.9047 | 0.9344 | 0.9193 |

## Inference Timing — Overall
### End-to-End (wall-clock, includes pre/post-process + Python overhead)
- **Mean:** 57.4 ms  (17.41 FPS)
- **Median:** 24.0 ms
- **Std Dev:** 60.5 ms
- **Mean (excl. first image / warm-up):** 53.0 ms (18.89 FPS)

### Pure Neural-Network Inference (Ultralytics `result.speed['inference']`)
- **Mean:** 21.6 ms  (46.23 FPS)
- **Median:** 16.1 ms
- **Std Dev:** 15.7 ms
- **Mean (excl. first image / warm-up):** 21.7 ms (46.15 FPS)

## Inference Timing — By Class Present in Image
_Note: this groups images by which ground-truth classes they contain; it is the mean total inference time for images containing that class, not a per-detection or per-class model cost (YOLO predicts all classes in a single forward pass per image)._

| Class Code | Mean Image Time (ms) | Images (n) |
|--- |--- |--- |
| BROST | 71.7 | 39 |
| STEME | 48.9 | 37 |
| URTUR | 48.3 | 31 |

## Setup Configuration Context
- **Model Identifier:** `yolov8m_ep150_b8_lr0.001`
- **Eval Platform:** `desktop`
- **Device Used:** `0`
- **Data Type:** `float32`
