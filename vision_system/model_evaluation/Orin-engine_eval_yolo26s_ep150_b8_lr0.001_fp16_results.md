# Evaluation Report: YOLO (yolo26s_ep150_b8_lr0.001_fp16)

## Core Metrics Summary
- **Precision:** 0.8562
- **Recall:** 0.8483
- **F1 Score:** 0.8522
- **Mean IoU (Matched):** 0.8439

## Bounding Box Breakdown
- **Total Ground Truth Boxes:** 2162
- **Total Predicted Boxes:** 2142
- **True Positives (TP):** 1834
- **False Positives (FP):** 308
- **False Negatives (FN):** 328

## Per-Class Metrics Table
| Class Code | TP | FP | FN | Precision | Recall | F1 Score |
|--- |--- |--- |--- |--- |--- |--- |
| URTUR | 827 | 78 | 87 | 0.9138 | 0.9048 | 0.9093 |
| STEME | 765 | 192 | 186 | 0.7994 | 0.8044 | 0.8019 |
| BROST | 242 | 38 | 55 | 0.8643 | 0.8148 | 0.8388 |

## Inference Timing — Overall
- **Mean:** 103.2 ms  (9.69 FPS)
- **Median:** 48.8 ms
- **Std Dev:** 240.2 ms
- **Mean (excl. first image / warm-up):** 80.5 ms (12.43 FPS)

## Inference Timing — By Class Present in Image
_Note: this groups images by which ground-truth classes they contain; it is the mean total inference time for images containing that class, not a per-detection or per-class model cost (YOLO predicts all classes in a single forward pass per image)._

| Class Code | Mean Image Time (ms) | Images (n) |
|--- |--- |--- |
| STEME | 148.4 | 37 |
| URTUR | 82.7 | 31 |
| BROST | 75.7 | 39 |

## Setup Configuration Context
- **Model Identifier:** `yolo26s_ep150_b8_lr0.001_fp16`
- **Eval Platform:** `Orin-engine`
- **Device Used:** `0`
- **Data Type:** `float32`
