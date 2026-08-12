# Evaluation Report: YOLO (yolo26s_ep150_b8_lr0.001)

## Core Metrics Summary
- **Precision:** 0.8712
- **Recall:** 0.8700
- **F1 Score:** 0.8706
- **Mean IoU (Matched):** 0.8403

## Bounding Box Breakdown
- **Total Ground Truth Boxes:** 2162
- **Total Predicted Boxes:** 2159
- **True Positives (TP):** 1881
- **False Positives (FP):** 278
- **False Negatives (FN):** 281

## Per-Class Metrics Table
| Class Code | TP | FP | FN | Precision | Recall | F1 Score |
|--- |--- |--- |--- |--- |--- |--- |
| STEME | 789 | 167 | 162 | 0.8253 | 0.8297 | 0.8275 |
| URTUR | 844 | 70 | 70 | 0.9234 | 0.9234 | 0.9234 |
| BROST | 248 | 41 | 49 | 0.8581 | 0.8350 | 0.8464 |

## Inference Timing — Overall
### End-to-End (wall-clock, includes pre/post-process + Python overhead)
- **Mean:** 104.1 ms  (9.60 FPS)
- **Median:** 45.4 ms
- **Std Dev:** 132.9 ms
- **Mean (excl. first image / warm-up):** 92.8 ms (10.78 FPS)

### Pure Neural-Network Inference (Ultralytics `result.speed['inference']`)
- **Mean:** 31.9 ms  (31.33 FPS)
- **Median:** 28.0 ms
- **Std Dev:** 15.4 ms
- **Mean (excl. first image / warm-up):** 30.6 ms (32.70 FPS)

## Inference Timing — By Class Present in Image
_Note: this groups images by which ground-truth classes they contain; it is the mean total inference time for images containing that class, not a per-detection or per-class model cost (YOLO predicts all classes in a single forward pass per image)._

| Class Code | Mean Image Time (ms) | Images (n) |
|--- |--- |--- |
| STEME | 130.9 | 37 |
| URTUR | 96.5 | 31 |
| BROST | 83.4 | 39 |

## Setup Configuration Context
- **Model Identifier:** `yolo26s_ep150_b8_lr0.001`
- **Eval Platform:** `Orin`
- **Device Used:** `0`
- **Data Type:** `float32`
