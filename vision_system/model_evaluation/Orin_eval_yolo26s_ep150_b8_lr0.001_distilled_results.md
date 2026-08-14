# Evaluation Report: YOLO (yolo26s_ep150_b8_lr0.001_distilled)

## Core Metrics Summary
- **Precision:** 0.8590
- **Recall:** 0.8649
- **F1 Score:** 0.8619
- **Mean IoU (Matched):** 0.8248

## Bounding Box Breakdown
- **Total Ground Truth Boxes:** 2162
- **Total Predicted Boxes:** 2177
- **True Positives (TP):** 1870
- **False Positives (FP):** 307
- **False Negatives (FN):** 292

## Per-Class Metrics Table
| Class Code | TP | FP | FN | Precision | Recall | F1 Score |
|--- |--- |--- |--- |--- |--- |--- |
| URTUR | 841 | 93 | 73 | 0.9004 | 0.9201 | 0.9102 |
| STEME | 788 | 170 | 163 | 0.8225 | 0.8286 | 0.8256 |
| BROST | 241 | 44 | 56 | 0.8456 | 0.8114 | 0.8282 |

## Inference Timing — Overall
### End-to-End (wall-clock, includes pre/post-process + Python overhead)
- **Mean:** 108.6 ms  (9.21 FPS)
- **Median:** 57.0 ms
- **Std Dev:** 134.1 ms
- **Mean (excl. first image / warm-up):** 97.2 ms (10.29 FPS)

### Pure Neural-Network Inference (Ultralytics `result.speed['inference']`)
- **Mean:** 36.2 ms  (27.59 FPS)
- **Median:** 32.2 ms
- **Std Dev:** 16.4 ms
- **Mean (excl. first image / warm-up):** 34.9 ms (28.66 FPS)

## Inference Timing — By Class Present in Image
_Note: this groups images by which ground-truth classes they contain; it is the mean total inference time for images containing that class, not a per-detection or per-class model cost (YOLO predicts all classes in a single forward pass per image)._

| Class Code | Mean Image Time (ms) | Images (n) |
|--- |--- |--- |
| STEME | 130.2 | 37 |
| URTUR | 105.5 | 31 |
| BROST | 89.0 | 39 |

## Setup Configuration Context
- **Model Identifier:** `yolo26s_ep150_b8_lr0.001_distilled`
- **Eval Platform:** `Orin`
- **Device Used:** `0`
- **Data Type:** `float16`
