# Evaluation Report: YOLO (yolo11s_ep150_b8_lr0.001_distilled_fp16)

## Core Metrics Summary
- **Precision:** 0.8435
- **Recall:** 0.8552
- **F1 Score:** 0.8493
- **Mean IoU (Matched):** 0.8338

## Bounding Box Breakdown
- **Total Ground Truth Boxes:** 2162
- **Total Predicted Boxes:** 2192
- **True Positives (TP):** 1849
- **False Positives (FP):** 343
- **False Negatives (FN):** 313

## Per-Class Metrics Table
| Class Code | TP | FP | FN | Precision | Recall | F1 Score |
|--- |--- |--- |--- |--- |--- |--- |
| STEME | 786 | 200 | 165 | 0.7972 | 0.8265 | 0.8116 |
| URTUR | 816 | 78 | 98 | 0.9128 | 0.8928 | 0.9027 |
| BROST | 247 | 65 | 50 | 0.7917 | 0.8316 | 0.8112 |

## Inference Timing — Overall
### End-to-End (wall-clock, includes pre/post-process + Python overhead)
- **Mean:** 105.9 ms  (9.44 FPS)
- **Median:** 48.8 ms
- **Std Dev:** 208.8 ms
- **Mean (excl. first image / warm-up):** 86.4 ms (11.57 FPS)

### Pure Neural-Network Inference (Ultralytics `result.speed['inference']`)
- **Mean:** 18.2 ms  (54.89 FPS)
- **Median:** 16.3 ms
- **Std Dev:** 6.3 ms
- **Mean (excl. first image / warm-up):** 18.2 ms (54.81 FPS)

## Inference Timing — By Class Present in Image
_Note: this groups images by which ground-truth classes they contain; it is the mean total inference time for images containing that class, not a per-detection or per-class model cost (YOLO predicts all classes in a single forward pass per image)._

| Class Code | Mean Image Time (ms) | Images (n) |
|--- |--- |--- |
| STEME | 146.8 | 37 |
| URTUR | 91.6 | 31 |
| BROST | 77.8 | 39 |

## Setup Configuration Context
- **Model Identifier:** `yolo11s_ep150_b8_lr0.001_distilled_fp16`
- **Eval Platform:** `Orin-engine`
- **Device Used:** `0`
- **Data Type:** `float16`
