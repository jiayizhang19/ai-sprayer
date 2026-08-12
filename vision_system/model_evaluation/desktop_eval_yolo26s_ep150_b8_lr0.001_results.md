# Evaluation Report: YOLO (yolo26s_ep150_b8_lr0.001)

## Core Metrics Summary
- **Precision:** 0.8721
- **Recall:** 0.8705
- **F1 Score:** 0.8713
- **Mean IoU (Matched):** 0.8402

## Bounding Box Breakdown
- **Total Ground Truth Boxes:** 2162
- **Total Predicted Boxes:** 2158
- **True Positives (TP):** 1882
- **False Positives (FP):** 276
- **False Negatives (FN):** 280

## Per-Class Metrics Table
| Class Code | TP | FP | FN | Precision | Recall | F1 Score |
|--- |--- |--- |--- |--- |--- |--- |
| BROST | 248 | 40 | 49 | 0.8611 | 0.8350 | 0.8479 |
| STEME | 790 | 166 | 161 | 0.8264 | 0.8307 | 0.8285 |
| URTUR | 844 | 70 | 70 | 0.9234 | 0.9234 | 0.9234 |

## Inference Timing — Overall
### End-to-End (wall-clock, includes pre/post-process + Python overhead)
- **Mean:** 45.3 ms  (22.08 FPS)
- **Median:** 14.8 ms
- **Std Dev:** 60.6 ms
- **Mean (excl. first image / warm-up):** 40.4 ms (24.76 FPS)

### Pure Neural-Network Inference (Ultralytics `result.speed['inference']`)
- **Mean:** 10.6 ms  (94.47 FPS)
- **Median:** 7.8 ms
- **Std Dev:** 10.3 ms
- **Mean (excl. first image / warm-up):** 10.6 ms (94.25 FPS)

## Inference Timing — By Class Present in Image
_Note: this groups images by which ground-truth classes they contain; it is the mean total inference time for images containing that class, not a per-detection or per-class model cost (YOLO predicts all classes in a single forward pass per image)._

| Class Code | Mean Image Time (ms) | Images (n) |
|--- |--- |--- |
| BROST | 55.1 | 39 |
| STEME | 40.5 | 37 |
| URTUR | 38.0 | 31 |

## Setup Configuration Context
- **Model Identifier:** `yolo26s_ep150_b8_lr0.001`
- **Eval Platform:** `desktop`
- **Device Used:** `0`
- **Data Type:** `float32`
