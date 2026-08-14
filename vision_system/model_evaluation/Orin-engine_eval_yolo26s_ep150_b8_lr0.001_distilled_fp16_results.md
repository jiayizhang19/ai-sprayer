# Evaluation Report: YOLO (yolo26s_ep150_b8_lr0.001_distilled_fp16)

## Core Metrics Summary
- **Precision:** 0.8558
- **Recall:** 0.8455
- **F1 Score:** 0.8506
- **Mean IoU (Matched):** 0.8368

## Bounding Box Breakdown
- **Total Ground Truth Boxes:** 2162
- **Total Predicted Boxes:** 2136
- **True Positives (TP):** 1828
- **False Positives (FP):** 308
- **False Negatives (FN):** 334

## Per-Class Metrics Table
| Class Code | TP | FP | FN | Precision | Recall | F1 Score |
|--- |--- |--- |--- |--- |--- |--- |
| URTUR | 831 | 113 | 83 | 0.8803 | 0.9092 | 0.8945 |
| STEME | 764 | 159 | 187 | 0.8277 | 0.8034 | 0.8154 |
| BROST | 233 | 36 | 64 | 0.8662 | 0.7845 | 0.8233 |

## Inference Timing — Overall
### End-to-End (wall-clock, includes pre/post-process + Python overhead)
- **Mean:** 89.3 ms  (11.20 FPS)
- **Median:** 41.2 ms
- **Std Dev:** 83.1 ms
- **Mean (excl. first image / warm-up):** 83.4 ms (11.99 FPS)

### Pure Neural-Network Inference (Ultralytics `result.speed['inference']`)
- **Mean:** 18.6 ms  (53.67 FPS)
- **Median:** 16.7 ms
- **Std Dev:** 5.4 ms
- **Mean (excl. first image / warm-up):** 18.6 ms (53.89 FPS)

## Inference Timing — By Class Present in Image
_Note: this groups images by which ground-truth classes they contain; it is the mean total inference time for images containing that class, not a per-detection or per-class model cost (YOLO predicts all classes in a single forward pass per image)._

| Class Code | Mean Image Time (ms) | Images (n) |
|--- |--- |--- |
| STEME | 103.8 | 37 |
| URTUR | 85.1 | 31 |
| BROST | 77.7 | 39 |

## Setup Configuration Context
- **Model Identifier:** `yolo26s_ep150_b8_lr0.001_distilled_fp16`
- **Eval Platform:** `Orin-engine`
- **Device Used:** `0`
- **Data Type:** `float16`
