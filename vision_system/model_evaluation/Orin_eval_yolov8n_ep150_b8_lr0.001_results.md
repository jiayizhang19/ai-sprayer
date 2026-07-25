# Evaluation Report: YOLO (yolov8n_ep150_b8_lr0.001)

## Core Metrics Summary
- **Precision:** 0.8245
- **Recall:** 0.8608
- **F1 Score:** 0.8423
- **Mean IoU (Matched):** 0.8340

## Bounding Box Breakdown
- **Total Ground Truth Boxes:** 2162
- **Total Predicted Boxes:** 2257
- **True Positives (TP):** 1861
- **False Positives (FP):** 396
- **False Negatives (FN):** 301

## Per-Class Metrics Table
| Class Code | TP | FP | FN | Precision | Recall | F1 Score |
|--- |--- |--- |--- |--- |--- |--- |
| STEME | 790 | 259 | 161 | 0.7531 | 0.8307 | 0.7900 |
| URTUR | 832 | 71 | 82 | 0.9214 | 0.9103 | 0.9158 |
| BROST | 239 | 66 | 58 | 0.7836 | 0.8047 | 0.7940 |

## Inference Timing — Overall
- **Mean:** 113.2 ms  (8.84 FPS)
- **Median:** 52.3 ms
- **Std Dev:** 222.9 ms
- **Mean (excl. first image / warm-up):** 92.4 ms (10.82 FPS)

## Inference Timing — By Class Present in Image
_Note: this groups images by which ground-truth classes they contain; it is the mean total inference time for images containing that class, not a per-detection or per-class model cost (YOLO predicts all classes in a single forward pass per image)._

| Class Code | Mean Image Time (ms) | Images (n) |
|--- |--- |--- |
| STEME | 153.7 | 37 |
| URTUR | 92.8 | 31 |
| BROST | 89.9 | 39 |

## Setup Configuration Context
- **Model Identifier:** `yolov8n_ep150_b8_lr0.001`
- **Eval Platform:** `Orin`
- **Device Used:** `0`
- **Data Type:** `float32`
