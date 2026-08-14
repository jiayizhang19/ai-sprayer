# Evaluation Report: YOLO (yolov10s_ep150_b8_lr0.001_fp16)

## Core Metrics Summary
- **Precision:** 0.8399
- **Recall:** 0.8520
- **F1 Score:** 0.8459
- **Mean IoU (Matched):** 0.8270

## Bounding Box Breakdown
- **Total Ground Truth Boxes:** 2162
- **Total Predicted Boxes:** 2193
- **True Positives (TP):** 1842
- **False Positives (FP):** 351
- **False Negatives (FN):** 320

## Per-Class Metrics Table
| Class Code | TP | FP | FN | Precision | Recall | F1 Score |
|--- |--- |--- |--- |--- |--- |--- |
| STEME | 768 | 202 | 183 | 0.7918 | 0.8076 | 0.7996 |
| URTUR | 831 | 109 | 83 | 0.8840 | 0.9092 | 0.8964 |
| BROST | 243 | 40 | 54 | 0.8587 | 0.8182 | 0.8379 |

## Inference Timing — Overall
### End-to-End (wall-clock, includes pre/post-process + Python overhead)
- **Mean:** 90.0 ms  (11.11 FPS)
- **Median:** 40.6 ms
- **Std Dev:** 85.3 ms
- **Mean (excl. first image / warm-up):** 84.1 ms (11.89 FPS)

### Pure Neural-Network Inference (Ultralytics `result.speed['inference']`)
- **Mean:** 17.5 ms  (57.20 FPS)
- **Median:** 16.7 ms
- **Std Dev:** 3.8 ms
- **Mean (excl. first image / warm-up):** 17.5 ms (57.14 FPS)

## Inference Timing — By Class Present in Image
_Note: this groups images by which ground-truth classes they contain; it is the mean total inference time for images containing that class, not a per-detection or per-class model cost (YOLO predicts all classes in a single forward pass per image)._

| Class Code | Mean Image Time (ms) | Images (n) |
|--- |--- |--- |
| STEME | 105.2 | 37 |
| URTUR | 87.0 | 31 |
| BROST | 76.2 | 39 |

## Setup Configuration Context
- **Model Identifier:** `yolov10s_ep150_b8_lr0.001_fp16`
- **Eval Platform:** `Orin-engine`
- **Device Used:** `0`
- **Data Type:** `float16`
