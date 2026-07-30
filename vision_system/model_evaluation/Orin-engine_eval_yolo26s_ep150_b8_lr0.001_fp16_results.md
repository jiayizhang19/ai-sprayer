# Evaluation Report: YOLO (yolo26s_ep150_b8_lr0.001_fp16)

## Core Metrics Summary
- **Precision:** 0.8586
- **Recall:** 0.8483
- **F1 Score:** 0.8534
- **Mean IoU (Matched):** 0.8438

## Bounding Box Breakdown
- **Total Ground Truth Boxes:** 2162
- **Total Predicted Boxes:** 2136
- **True Positives (TP):** 1834
- **False Positives (FP):** 302
- **False Negatives (FN):** 328

## Per-Class Metrics Table
| Class Code | TP | FP | FN | Precision | Recall | F1 Score |
|--- |--- |--- |--- |--- |--- |--- |
| URTUR | 827 | 76 | 87 | 0.9158 | 0.9048 | 0.9103 |
| STEME | 765 | 189 | 186 | 0.8019 | 0.8044 | 0.8031 |
| BROST | 242 | 37 | 55 | 0.8674 | 0.8148 | 0.8403 |

## Inference Timing — Overall
### End-to-End (wall-clock, includes pre/post-process + Python overhead)
- **Mean:** 81.2 ms  (12.31 FPS)
- **Median:** 30.2 ms
- **Std Dev:** 92.8 ms
- **Mean (excl. first image / warm-up):** 74.4 ms (13.44 FPS)

### Pure Neural-Network Inference (Ultralytics `result.speed['inference']`)
- **Mean:** 13.5 ms  (74.08 FPS)
- **Median:** 11.8 ms
- **Std Dev:** 6.9 ms
- **Mean (excl. first image / warm-up):** 13.6 ms (73.79 FPS)

## Inference Timing — By Class Present in Image
_Note: this groups images by which ground-truth classes they contain; it is the mean total inference time for images containing that class, not a per-detection or per-class model cost (YOLO predicts all classes in a single forward pass per image)._

| Class Code | Mean Image Time (ms) | Images (n) |
|--- |--- |--- |
| STEME | 94.8 | 37 |
| URTUR | 75.7 | 31 |
| BROST | 70.9 | 39 |

## Setup Configuration Context
- **Model Identifier:** `yolo26s_ep150_b8_lr0.001_fp16`
- **Eval Platform:** `Orin-engine`
- **Device Used:** `0`
- **Data Type:** `float32`
