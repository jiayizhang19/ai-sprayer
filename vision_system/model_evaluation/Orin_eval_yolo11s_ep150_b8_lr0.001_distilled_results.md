# Evaluation Report: YOLO (yolo11s_ep150_b8_lr0.001_distilled)

## Core Metrics Summary
- **Precision:** 0.8525
- **Recall:** 0.8742
- **F1 Score:** 0.8632
- **Mean IoU (Matched):** 0.8288

## Bounding Box Breakdown
- **Total Ground Truth Boxes:** 2162
- **Total Predicted Boxes:** 2217
- **True Positives (TP):** 1890
- **False Positives (FP):** 327
- **False Negatives (FN):** 272

## Per-Class Metrics Table
| Class Code | TP | FP | FN | Precision | Recall | F1 Score |
|--- |--- |--- |--- |--- |--- |--- |
| STEME | 802 | 193 | 149 | 0.8060 | 0.8433 | 0.8243 |
| URTUR | 835 | 78 | 79 | 0.9146 | 0.9136 | 0.9141 |
| BROST | 253 | 56 | 44 | 0.8188 | 0.8519 | 0.8350 |

## Inference Timing — Overall
### End-to-End (wall-clock, includes pre/post-process + Python overhead)
- **Mean:** 123.1 ms  (8.12 FPS)
- **Median:** 46.5 ms
- **Std Dev:** 273.4 ms
- **Mean (excl. first image / warm-up):** 97.5 ms (10.26 FPS)

### Pure Neural-Network Inference (Ultralytics `result.speed['inference']`)
- **Mean:** 32.9 ms  (30.40 FPS)
- **Median:** 26.8 ms
- **Std Dev:** 16.8 ms
- **Mean (excl. first image / warm-up):** 31.6 ms (31.65 FPS)

## Inference Timing — By Class Present in Image
_Note: this groups images by which ground-truth classes they contain; it is the mean total inference time for images containing that class, not a per-detection or per-class model cost (YOLO predicts all classes in a single forward pass per image)._

| Class Code | Mean Image Time (ms) | Images (n) |
|--- |--- |--- |
| STEME | 173.7 | 37 |
| URTUR | 104.0 | 31 |
| BROST | 89.1 | 39 |

## Setup Configuration Context
- **Model Identifier:** `yolo11s_ep150_b8_lr0.001_distilled`
- **Eval Platform:** `Orin`
- **Device Used:** `0`
- **Data Type:** `float16`
