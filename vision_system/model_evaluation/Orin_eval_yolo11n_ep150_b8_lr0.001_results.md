# Evaluation Report: YOLO (yolo11n_ep150_b8_lr0.001)

## Core Metrics Summary
- **Precision:** 0.8252
- **Recall:** 0.8497
- **F1 Score:** 0.8373
- **Mean IoU (Matched):** 0.8385

## Bounding Box Breakdown
- **Total Ground Truth Boxes:** 2162
- **Total Predicted Boxes:** 2226
- **True Positives (TP):** 1837
- **False Positives (FP):** 389
- **False Negatives (FN):** 325

## Per-Class Metrics Table
| Class Code | TP | FP | FN | Precision | Recall | F1 Score |
|--- |--- |--- |--- |--- |--- |--- |
| STEME | 781 | 246 | 170 | 0.7605 | 0.8212 | 0.7897 |
| URTUR | 820 | 92 | 94 | 0.8991 | 0.8972 | 0.8981 |
| BROST | 236 | 51 | 61 | 0.8223 | 0.7946 | 0.8082 |

## Inference Timing — Overall
### End-to-End (wall-clock, includes pre/post-process + Python overhead)
- **Mean:** 115.1 ms  (8.69 FPS)
- **Median:** 45.8 ms
- **Std Dev:** 261.3 ms
- **Mean (excl. first image / warm-up):** 90.5 ms (11.05 FPS)

### Pure Neural-Network Inference (Ultralytics `result.speed['inference']`)
- **Mean:** 26.4 ms  (37.87 FPS)
- **Median:** 23.3 ms
- **Std Dev:** 11.9 ms
- **Mean (excl. first image / warm-up):** 25.4 ms (39.30 FPS)

## Inference Timing — By Class Present in Image
_Note: this groups images by which ground-truth classes they contain; it is the mean total inference time for images containing that class, not a per-detection or per-class model cost (YOLO predicts all classes in a single forward pass per image)._

| Class Code | Mean Image Time (ms) | Images (n) |
|--- |--- |--- |
| STEME | 166.2 | 37 |
| URTUR | 91.7 | 31 |
| BROST | 84.5 | 39 |

## Setup Configuration Context
- **Model Identifier:** `yolo11n_ep150_b8_lr0.001`
- **Eval Platform:** `Orin`
- **Device Used:** `0`
- **Data Type:** `float32`
