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
- **Mean:** 107.2 ms  (9.32 FPS)
- **Median:** 50.9 ms
- **Std Dev:** 113.3 ms
- **Mean (excl. first image / warm-up):** 98.4 ms (10.16 FPS)

## Inference Timing — By Class Present in Image
_Note: this groups images by which ground-truth classes they contain; it is the mean total inference time for images containing that class, not a per-detection or per-class model cost (YOLO predicts all classes in a single forward pass per image)._

| Class Code | Mean Image Time (ms) | Images (n) |
|--- |--- |--- |
| STEME | 125.0 | 37 |
| URTUR | 100.1 | 31 |
| BROST | 93.4 | 39 |

## Setup Configuration Context
- **Model Identifier:** `yolo26s_ep150_b8_lr0.001`
- **Eval Platform:** `Orin`
- **Device Used:** `0`
- **Data Type:** `float32`
