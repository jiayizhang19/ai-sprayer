# Evaluation Report: YOLO (yolov8m_ep150_b8_lr0.001)

## Core Metrics Summary
- **Precision:** 0.8646
- **Recall:** 0.8834
- **F1 Score:** 0.8739
- **Mean IoU (Matched):** 0.8314

## Bounding Box Breakdown
- **Total Ground Truth Boxes:** 2162
- **Total Predicted Boxes:** 2209
- **True Positives (TP):** 1910
- **False Positives (FP):** 299
- **False Negatives (FN):** 252

## Per-Class Metrics Table
| Class Code | TP | FP | FN | Precision | Recall | F1 Score |
|--- |--- |--- |--- |--- |--- |--- |
| URTUR | 854 | 90 | 60 | 0.9047 | 0.9344 | 0.9193 |
| STEME | 806 | 150 | 145 | 0.8431 | 0.8475 | 0.8453 |
| BROST | 250 | 59 | 47 | 0.8091 | 0.8418 | 0.8251 |

## Inference Timing — Overall
### End-to-End (wall-clock, includes pre/post-process + Python overhead)
- **Mean:** 144.5 ms  (6.92 FPS)
- **Median:** 80.2 ms
- **Std Dev:** 259.7 ms
- **Mean (excl. first image / warm-up):** 120.1 ms (8.33 FPS)

### Pure Neural-Network Inference (Ultralytics `result.speed['inference']`)
- **Mean:** 54.8 ms  (18.26 FPS)
- **Median:** 52.6 ms
- **Std Dev:** 11.6 ms
- **Mean (excl. first image / warm-up):** 53.9 ms (18.54 FPS)

## Inference Timing — By Class Present in Image
_Note: this groups images by which ground-truth classes they contain; it is the mean total inference time for images containing that class, not a per-detection or per-class model cost (YOLO predicts all classes in a single forward pass per image)._

| Class Code | Mean Image Time (ms) | Images (n) |
|--- |--- |--- |
| STEME | 194.6 | 37 |
| URTUR | 122.0 | 31 |
| BROST | 114.4 | 39 |

## Setup Configuration Context
- **Model Identifier:** `yolov8m_ep150_b8_lr0.001`
- **Eval Platform:** `Orin`
- **Device Used:** `0`
- **Data Type:** `float32`
