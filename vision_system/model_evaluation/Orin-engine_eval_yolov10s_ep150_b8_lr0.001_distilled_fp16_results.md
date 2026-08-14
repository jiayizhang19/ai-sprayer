# Evaluation Report: YOLO (yolov10s_ep150_b8_lr0.001_distilled_fp16)

## Core Metrics Summary
- **Precision:** 0.8499
- **Recall:** 0.8409
- **F1 Score:** 0.8454
- **Mean IoU (Matched):** 0.8313

## Bounding Box Breakdown
- **Total Ground Truth Boxes:** 2162
- **Total Predicted Boxes:** 2139
- **True Positives (TP):** 1818
- **False Positives (FP):** 321
- **False Negatives (FN):** 344

## Per-Class Metrics Table
| Class Code | TP | FP | FN | Precision | Recall | F1 Score |
|--- |--- |--- |--- |--- |--- |--- |
| URTUR | 824 | 106 | 90 | 0.8860 | 0.9015 | 0.8937 |
| STEME | 763 | 177 | 188 | 0.8117 | 0.8023 | 0.8070 |
| BROST | 231 | 38 | 66 | 0.8587 | 0.7778 | 0.8163 |

## Inference Timing — Overall
### End-to-End (wall-clock, includes pre/post-process + Python overhead)
- **Mean:** 87.3 ms  (11.45 FPS)
- **Median:** 40.0 ms
- **Std Dev:** 83.5 ms
- **Mean (excl. first image / warm-up):** 81.5 ms (12.26 FPS)

### Pure Neural-Network Inference (Ultralytics `result.speed['inference']`)
- **Mean:** 18.7 ms  (53.56 FPS)
- **Median:** 17.3 ms
- **Std Dev:** 5.5 ms
- **Mean (excl. first image / warm-up):** 18.7 ms (53.47 FPS)

## Inference Timing — By Class Present in Image
_Note: this groups images by which ground-truth classes they contain; it is the mean total inference time for images containing that class, not a per-detection or per-class model cost (YOLO predicts all classes in a single forward pass per image)._

| Class Code | Mean Image Time (ms) | Images (n) |
|--- |--- |--- |
| STEME | 101.8 | 37 |
| URTUR | 84.4 | 31 |
| BROST | 74.9 | 39 |

## Setup Configuration Context
- **Model Identifier:** `yolov10s_ep150_b8_lr0.001_distilled_fp16`
- **Eval Platform:** `Orin-engine`
- **Device Used:** `0`
- **Data Type:** `float16`
