# Evaluation Report: YOLO (yolo11s_ep150_b8_lr0.001_fp16)

## Core Metrics Summary
- **Precision:** 0.8205
- **Recall:** 0.8585
- **F1 Score:** 0.8391
- **Mean IoU (Matched):** 0.8317

## Bounding Box Breakdown
- **Total Ground Truth Boxes:** 2162
- **Total Predicted Boxes:** 2262
- **True Positives (TP):** 1856
- **False Positives (FP):** 406
- **False Negatives (FN):** 306

## Per-Class Metrics Table
| Class Code | TP | FP | FN | Precision | Recall | F1 Score |
|--- |--- |--- |--- |--- |--- |--- |
| URTUR | 826 | 76 | 88 | 0.9157 | 0.9037 | 0.9097 |
| STEME | 785 | 274 | 166 | 0.7413 | 0.8254 | 0.7811 |
| BROST | 245 | 56 | 52 | 0.8140 | 0.8249 | 0.8194 |

## Inference Timing — Overall
### End-to-End (wall-clock, includes pre/post-process + Python overhead)
- **Mean:** 94.4 ms  (10.59 FPS)
- **Median:** 28.8 ms
- **Std Dev:** 220.9 ms
- **Mean (excl. first image / warm-up):** 73.9 ms (13.54 FPS)

### Pure Neural-Network Inference (Ultralytics `result.speed['inference']`)
- **Mean:** 9.6 ms  (104.10 FPS)
- **Median:** 9.2 ms
- **Std Dev:** 3.3 ms
- **Mean (excl. first image / warm-up):** 9.6 ms (104.46 FPS)

## Inference Timing — By Class Present in Image
_Note: this groups images by which ground-truth classes they contain; it is the mean total inference time for images containing that class, not a per-detection or per-class model cost (YOLO predicts all classes in a single forward pass per image)._

| Class Code | Mean Image Time (ms) | Images (n) |
|--- |--- |--- |
| STEME | 139.2 | 37 |
| URTUR | 77.6 | 31 |
| BROST | 64.9 | 39 |

## Setup Configuration Context
- **Model Identifier:** `yolo11s_ep150_b8_lr0.001_fp16`
- **Eval Platform:** `Orin-engine`
- **Device Used:** `0`
- **Data Type:** `float16`
