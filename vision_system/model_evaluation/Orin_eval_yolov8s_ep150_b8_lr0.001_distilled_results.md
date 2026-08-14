# Evaluation Report: YOLO (yolov8s_ep150_b8_lr0.001_distilled)

## Core Metrics Summary
- **Precision:** 0.8724
- **Recall:** 0.8636
- **F1 Score:** 0.8680
- **Mean IoU (Matched):** 0.8374

## Bounding Box Breakdown
- **Total Ground Truth Boxes:** 2162
- **Total Predicted Boxes:** 2140
- **True Positives (TP):** 1867
- **False Positives (FP):** 273
- **False Negatives (FN):** 295

## Per-Class Metrics Table
| Class Code | TP | FP | FN | Precision | Recall | F1 Score |
|--- |--- |--- |--- |--- |--- |--- |
| URTUR | 845 | 80 | 69 | 0.9135 | 0.9245 | 0.9190 |
| STEME | 775 | 150 | 176 | 0.8378 | 0.8149 | 0.8262 |
| BROST | 247 | 43 | 50 | 0.8517 | 0.8316 | 0.8416 |

## Inference Timing — Overall
### End-to-End (wall-clock, includes pre/post-process + Python overhead)
- **Mean:** 117.8 ms  (8.49 FPS)
- **Median:** 44.7 ms
- **Std Dev:** 254.9 ms
- **Mean (excl. first image / warm-up):** 94.0 ms (10.64 FPS)

### Pure Neural-Network Inference (Ultralytics `result.speed['inference']`)
- **Mean:** 30.8 ms  (32.42 FPS)
- **Median:** 27.1 ms
- **Std Dev:** 13.3 ms
- **Mean (excl. first image / warm-up):** 29.9 ms (33.49 FPS)

## Inference Timing — By Class Present in Image
_Note: this groups images by which ground-truth classes they contain; it is the mean total inference time for images containing that class, not a per-detection or per-class model cost (YOLO predicts all classes in a single forward pass per image)._

| Class Code | Mean Image Time (ms) | Images (n) |
|--- |--- |--- |
| STEME | 165.7 | 37 |
| URTUR | 96.8 | 31 |
| BROST | 88.2 | 39 |

## Setup Configuration Context
- **Model Identifier:** `yolov8s_ep150_b8_lr0.001_distilled`
- **Eval Platform:** `Orin`
- **Device Used:** `0`
- **Data Type:** `float16`
