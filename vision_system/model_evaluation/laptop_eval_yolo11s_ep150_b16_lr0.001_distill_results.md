# Evaluation Report: YOLO (yolo11s_ep150_b16_lr0.001_distill)

## Core Metrics Summary
- **Precision:** 0.8525
- **Recall:** 0.8742
- **F1 Score:** 0.8632
- **Mean IoU (Matched):** 0.8287

## Bounding Box Breakdown
- **Total Ground Truth Boxes:** 2162
- **Total Predicted Boxes:** 2217
- **True Positives (TP):** 1890
- **False Positives (FP):** 327
- **False Negatives (FN):** 272

## Per-Class Metrics Table
| Class Code | TP | FP | FN | Precision | Recall | F1 Score |
|--- |--- |--- |--- |--- |--- |--- |
| BROST | 253 | 55 | 44 | 0.8214 | 0.8519 | 0.8364 |
| STEME | 802 | 194 | 149 | 0.8052 | 0.8433 | 0.8238 |
| URTUR | 835 | 78 | 79 | 0.9146 | 0.9136 | 0.9141 |

## Inference Timing — Overall
### End-to-End (wall-clock, includes pre/post-process + Python overhead)
- **Mean:** 152.0 ms  (6.58 FPS)
- **Median:** 122.9 ms
- **Std Dev:** 193.0 ms
- **Mean (excl. first image / warm-up):** 133.4 ms (7.49 FPS)

### Pure Neural-Network Inference (Ultralytics `result.speed['inference']`)
- **Mean:** 97.6 ms  (10.25 FPS)
- **Median:** 104.9 ms
- **Std Dev:** 16.0 ms
- **Mean (excl. first image / warm-up):** 97.3 ms (10.28 FPS)

## Inference Timing — By Class Present in Image
_Note: this groups images by which ground-truth classes they contain; it is the mean total inference time for images containing that class, not a per-detection or per-class model cost (YOLO predicts all classes in a single forward pass per image)._

| Class Code | Mean Image Time (ms) | Images (n) |
|--- |--- |--- |
| BROST | 183.3 | 39 |
| STEME | 133.1 | 37 |
| URTUR | 134.8 | 31 |

## Setup Configuration Context
- **Model Identifier:** `yolo11s_ep150_b16_lr0.001_distill`
- **Eval Platform:** `laptop`
- **Device Used:** `cpu`
- **Data Type:** `float32`
