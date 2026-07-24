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
- **Mean:** 196.5 ms  (5.09 FPS)
- **Median:** 192.9 ms
- **Std Dev:** 93.4 ms
- **Mean (excl. first image / warm-up):** 188.9 ms (5.29 FPS)

## Inference Timing — By Class Present in Image
_Note: this groups images by which ground-truth classes they contain; it is the mean total inference time for images containing that class, not a per-detection or per-class model cost (YOLO predicts all classes in a single forward pass per image)._

| Class Code | Mean Image Time (ms) | Images (n) |
|--- |--- |--- |
| BROST | 199.5 | 39 |
| STEME | 188.4 | 37 |
| URTUR | 200.3 | 31 |

## Setup Configuration Context
- **Model Identifier:** `yolov8m_ep150_b8_lr0.001`
- **Eval Platform:** `desktop`
- **Device Used:** `0`
- **Data Type:** `float32`
