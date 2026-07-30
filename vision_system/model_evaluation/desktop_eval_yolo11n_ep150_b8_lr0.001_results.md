# Evaluation Report: YOLO (yolo11n_ep150_b8_lr0.001)

## Core Metrics Summary
- **Precision:** 0.8252
- **Recall:** 0.8497
- **F1 Score:** 0.8373
- **Mean IoU (Matched):** 0.8384

## Bounding Box Breakdown
- **Total Ground Truth Boxes:** 2162
- **Total Predicted Boxes:** 2226
- **True Positives (TP):** 1837
- **False Positives (FP):** 389
- **False Negatives (FN):** 325

## Per-Class Metrics Table
| Class Code | TP | FP | FN | Precision | Recall | F1 Score |
|--- |--- |--- |--- |--- |--- |--- |
| BROST | 236 | 50 | 61 | 0.8252 | 0.7946 | 0.8096 |
| STEME | 781 | 247 | 170 | 0.7597 | 0.8212 | 0.7893 |
| URTUR | 820 | 92 | 94 | 0.8991 | 0.8972 | 0.8981 |

## Inference Timing — Overall
### End-to-End (wall-clock, includes pre/post-process + Python overhead)
- **Mean:** 46.3 ms  (21.62 FPS)
- **Median:** 12.9 ms
- **Std Dev:** 55.9 ms
- **Mean (excl. first image / warm-up):** 42.2 ms (23.72 FPS)

### Pure Neural-Network Inference (Ultralytics `result.speed['inference']`)
- **Mean:** 8.3 ms  (120.03 FPS)
- **Median:** 5.7 ms
- **Std Dev:** 7.7 ms
- **Mean (excl. first image / warm-up):** 8.3 ms (121.00 FPS)

## Inference Timing — By Class Present in Image
_Note: this groups images by which ground-truth classes they contain; it is the mean total inference time for images containing that class, not a per-detection or per-class model cost (YOLO predicts all classes in a single forward pass per image)._

| Class Code | Mean Image Time (ms) | Images (n) |
|--- |--- |--- |
| BROST | 53.4 | 39 |
| STEME | 42.4 | 37 |
| URTUR | 40.7 | 31 |

## Setup Configuration Context
- **Model Identifier:** `yolo11n_ep150_b8_lr0.001`
- **Eval Platform:** `desktop`
- **Device Used:** `0`
- **Data Type:** `float32`
