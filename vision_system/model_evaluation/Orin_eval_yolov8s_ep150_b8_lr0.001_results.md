# Evaluation Report: YOLO (yolov8s_ep150_b8_lr0.001)

## Core Metrics Summary
- **Precision:** 0.8652
- **Recall:** 0.8756
- **F1 Score:** 0.8703
- **Mean IoU (Matched):** 0.8371

## Bounding Box Breakdown
- **Total Ground Truth Boxes:** 2162
- **Total Predicted Boxes:** 2188
- **True Positives (TP):** 1893
- **False Positives (FP):** 295
- **False Negatives (FN):** 269

## Per-Class Metrics Table
| Class Code | TP | FP | FN | Precision | Recall | F1 Score |
|--- |--- |--- |--- |--- |--- |--- |
| URTUR | 851 | 91 | 63 | 0.9034 | 0.9311 | 0.9170 |
| STEME | 802 | 157 | 149 | 0.8363 | 0.8433 | 0.8398 |
| BROST | 240 | 47 | 57 | 0.8362 | 0.8081 | 0.8219 |

## Inference Timing — Overall
### End-to-End (wall-clock, includes pre/post-process + Python overhead)
- **Mean:** 123.6 ms  (8.09 FPS)
- **Median:** 54.7 ms
- **Std Dev:** 248.5 ms
- **Mean (excl. first image / warm-up):** 100.5 ms (9.95 FPS)

### Pure Neural-Network Inference (Ultralytics `result.speed['inference']`)
- **Mean:** 34.3 ms  (29.16 FPS)
- **Median:** 29.2 ms
- **Std Dev:** 15.1 ms
- **Mean (excl. first image / warm-up):** 33.3 ms (30.04 FPS)

## Inference Timing — By Class Present in Image
_Note: this groups images by which ground-truth classes they contain; it is the mean total inference time for images containing that class, not a per-detection or per-class model cost (YOLO predicts all classes in a single forward pass per image)._

| Class Code | Mean Image Time (ms) | Images (n) |
|--- |--- |--- |
| STEME | 170.5 | 37 |
| URTUR | 101.3 | 31 |
| BROST | 96.2 | 39 |

## Setup Configuration Context
- **Model Identifier:** `yolov8s_ep150_b8_lr0.001`
- **Eval Platform:** `Orin`
- **Device Used:** `0`
- **Data Type:** `float32`
