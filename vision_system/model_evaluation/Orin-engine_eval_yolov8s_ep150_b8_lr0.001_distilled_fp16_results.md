# Evaluation Report: YOLO (yolov8s_ep150_b8_lr0.001_distilled_fp16)

## Core Metrics Summary
- **Precision:** 0.8730
- **Recall:** 0.8492
- **F1 Score:** 0.8610
- **Mean IoU (Matched):** 0.8391

## Bounding Box Breakdown
- **Total Ground Truth Boxes:** 2162
- **Total Predicted Boxes:** 2103
- **True Positives (TP):** 1836
- **False Positives (FP):** 267
- **False Negatives (FN):** 326

## Per-Class Metrics Table
| Class Code | TP | FP | FN | Precision | Recall | F1 Score |
|--- |--- |--- |--- |--- |--- |--- |
| URTUR | 829 | 81 | 85 | 0.9110 | 0.9070 | 0.9090 |
| STEME | 771 | 151 | 180 | 0.8362 | 0.8107 | 0.8233 |
| BROST | 236 | 35 | 61 | 0.8708 | 0.7946 | 0.8310 |

## Inference Timing — Overall
### End-to-End (wall-clock, includes pre/post-process + Python overhead)
- **Mean:** 105.5 ms  (9.48 FPS)
- **Median:** 48.1 ms
- **Std Dev:** 210.6 ms
- **Mean (excl. first image / warm-up):** 85.8 ms (11.66 FPS)

### Pure Neural-Network Inference (Ultralytics `result.speed['inference']`)
- **Mean:** 16.9 ms  (59.12 FPS)
- **Median:** 15.6 ms
- **Std Dev:** 4.9 ms
- **Mean (excl. first image / warm-up):** 17.0 ms (58.97 FPS)

## Inference Timing — By Class Present in Image
_Note: this groups images by which ground-truth classes they contain; it is the mean total inference time for images containing that class, not a per-detection or per-class model cost (YOLO predicts all classes in a single forward pass per image)._

| Class Code | Mean Image Time (ms) | Images (n) |
|--- |--- |--- |
| STEME | 148.0 | 37 |
| URTUR | 89.3 | 31 |
| BROST | 77.6 | 39 |

## Setup Configuration Context
- **Model Identifier:** `yolov8s_ep150_b8_lr0.001_distilled_fp16`
- **Eval Platform:** `Orin-engine`
- **Device Used:** `0`
- **Data Type:** `float16`
