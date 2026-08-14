# Evaluation Report: YOLO (yolo11s_ep150_b16_lr0.001_distill)

## Core Metrics Summary
- **Precision:** 0.8524
- **Recall:** 0.8737
- **F1 Score:** 0.8630
- **Mean IoU (Matched):** 0.8289

## Bounding Box Breakdown
- **Total Ground Truth Boxes:** 2162
- **Total Predicted Boxes:** 2216
- **True Positives (TP):** 1889
- **False Positives (FP):** 327
- **False Negatives (FN):** 273

## Per-Class Metrics Table
| Class Code | TP | FP | FN | Precision | Recall | F1 Score |
|--- |--- |--- |--- |--- |--- |--- |
| BROST | 253 | 56 | 44 | 0.8188 | 0.8519 | 0.8350 |
| STEME | 802 | 193 | 149 | 0.8060 | 0.8433 | 0.8243 |
| URTUR | 834 | 78 | 80 | 0.9145 | 0.9125 | 0.9135 |

## Inference Timing — Overall
### End-to-End (wall-clock, includes pre/post-process + Python overhead)
- **Mean:** 76.2 ms  (13.13 FPS)
- **Median:** 14.8 ms
- **Std Dev:** 261.4 ms
- **Mean (excl. first image / warm-up):** 51.3 ms (19.48 FPS)

### Pure Neural-Network Inference (Ultralytics `result.speed['inference']`)
- **Mean:** 12.3 ms  (80.98 FPS)
- **Median:** 7.0 ms
- **Std Dev:** 13.9 ms
- **Mean (excl. first image / warm-up):** 12.4 ms (80.65 FPS)

## Inference Timing — By Class Present in Image
_Note: this groups images by which ground-truth classes they contain; it is the mean total inference time for images containing that class, not a per-detection or per-class model cost (YOLO predicts all classes in a single forward pass per image)._

| Class Code | Mean Image Time (ms) | Images (n) |
|--- |--- |--- |
| BROST | 117.8 | 39 |
| STEME | 55.0 | 37 |
| URTUR | 49.0 | 31 |

## Setup Configuration Context
- **Model Identifier:** `yolo11s_ep150_b16_lr0.001_distill`
- **Eval Platform:** `desktop`
- **Device Used:** `0`
- **Data Type:** `float32`
