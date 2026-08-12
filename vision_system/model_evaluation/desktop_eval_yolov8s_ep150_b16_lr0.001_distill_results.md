# Evaluation Report: YOLO (yolov8s_ep150_b16_lr0.001_distill)

## Core Metrics Summary
- **Precision:** 0.8724
- **Recall:** 0.8636
- **F1 Score:** 0.8680
- **Mean IoU (Matched):** 0.8375

## Bounding Box Breakdown
- **Total Ground Truth Boxes:** 2162
- **Total Predicted Boxes:** 2140
- **True Positives (TP):** 1867
- **False Positives (FP):** 273
- **False Negatives (FN):** 295

## Per-Class Metrics Table
| Class Code | TP | FP | FN | Precision | Recall | F1 Score |
|--- |--- |--- |--- |--- |--- |--- |
| BROST | 247 | 43 | 50 | 0.8517 | 0.8316 | 0.8416 |
| STEME | 775 | 150 | 176 | 0.8378 | 0.8149 | 0.8262 |
| URTUR | 845 | 80 | 69 | 0.9135 | 0.9245 | 0.9190 |

## Inference Timing — Overall
### End-to-End (wall-clock, includes pre/post-process + Python overhead)
- **Mean:** 75.6 ms  (13.23 FPS)
- **Median:** 31.3 ms
- **Std Dev:** 182.6 ms
- **Mean (excl. first image / warm-up):** 58.3 ms (17.16 FPS)

### Pure Neural-Network Inference (Ultralytics `result.speed['inference']`)
- **Mean:** 16.6 ms  (60.32 FPS)
- **Median:** 16.8 ms
- **Std Dev:** 5.2 ms
- **Mean (excl. first image / warm-up):** 16.6 ms (60.32 FPS)

## Inference Timing — By Class Present in Image
_Note: this groups images by which ground-truth classes they contain; it is the mean total inference time for images containing that class, not a per-detection or per-class model cost (YOLO predicts all classes in a single forward pass per image)._

| Class Code | Mean Image Time (ms) | Images (n) |
|--- |--- |--- |
| BROST | 101.6 | 39 |
| STEME | 61.8 | 37 |
| URTUR | 58.9 | 31 |

## Setup Configuration Context
- **Model Identifier:** `yolov8s_ep150_b16_lr0.001_distill`
- **Eval Platform:** `desktop`
- **Device Used:** `0`
- **Data Type:** `float32`
