# Evaluation Report: YOLO (yolo11s_ep150_b8_lr0.001)

## Core Metrics Summary
- **Precision:** 0.8369
- **Recall:** 0.8760
- **F1 Score:** 0.8560
- **Mean IoU (Matched):** 0.8281

## Bounding Box Breakdown
- **Total Ground Truth Boxes:** 2162
- **Total Predicted Boxes:** 2263
- **True Positives (TP):** 1894
- **False Positives (FP):** 369
- **False Negatives (FN):** 268

## Per-Class Metrics Table
| Class Code | TP | FP | FN | Precision | Recall | F1 Score |
|--- |--- |--- |--- |--- |--- |--- |
| URTUR | 841 | 81 | 73 | 0.9121 | 0.9201 | 0.9161 |
| STEME | 805 | 233 | 146 | 0.7755 | 0.8465 | 0.8095 |
| BROST | 248 | 55 | 49 | 0.8185 | 0.8350 | 0.8267 |

## Inference Timing — Overall
- **Mean:** 123.4 ms  (8.11 FPS)
- **Median:** 54.1 ms
- **Std Dev:** 264.0 ms
- **Mean (excl. first image / warm-up):** 98.7 ms (10.13 FPS)

## Inference Timing — By Class Present in Image
_Note: this groups images by which ground-truth classes they contain; it is the mean total inference time for images containing that class, not a per-detection or per-class model cost (YOLO predicts all classes in a single forward pass per image)._

| Class Code | Mean Image Time (ms) | Images (n) |
|--- |--- |--- |
| STEME | 171.1 | 37 |
| URTUR | 100.7 | 31 |
| BROST | 95.2 | 39 |

## Setup Configuration Context
- **Model Identifier:** `yolo11s_ep150_b8_lr0.001`
- **Eval Platform:** `Orin`
- **Device Used:** `0`
- **Data Type:** `float32`
