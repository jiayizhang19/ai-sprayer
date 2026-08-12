# Evaluation Report: YOLO (yolo11s_ep150_b8_lr0.001)

## Core Metrics Summary
- **Precision:** 0.8366
- **Recall:** 0.8760
- **F1 Score:** 0.8559
- **Mean IoU (Matched):** 0.8281

## Bounding Box Breakdown
- **Total Ground Truth Boxes:** 2162
- **Total Predicted Boxes:** 2264
- **True Positives (TP):** 1894
- **False Positives (FP):** 370
- **False Negatives (FN):** 268

## Per-Class Metrics Table
| Class Code | TP | FP | FN | Precision | Recall | F1 Score |
|--- |--- |--- |--- |--- |--- |--- |
| BROST | 248 | 55 | 49 | 0.8185 | 0.8350 | 0.8267 |
| STEME | 805 | 234 | 146 | 0.7748 | 0.8465 | 0.8090 |
| URTUR | 841 | 81 | 73 | 0.9121 | 0.9201 | 0.9161 |

## Inference Timing — Overall
### End-to-End (wall-clock, includes pre/post-process + Python overhead)
- **Mean:** 55.0 ms  (18.18 FPS)
- **Median:** 15.6 ms
- **Std Dev:** 66.0 ms
- **Mean (excl. first image / warm-up):** 50.4 ms (19.85 FPS)

### Pure Neural-Network Inference (Ultralytics `result.speed['inference']`)
- **Mean:** 12.2 ms  (82.19 FPS)
- **Median:** 6.8 ms
- **Std Dev:** 15.7 ms
- **Mean (excl. first image / warm-up):** 12.2 ms (81.87 FPS)

## Inference Timing — By Class Present in Image
_Note: this groups images by which ground-truth classes they contain; it is the mean total inference time for images containing that class, not a per-detection or per-class model cost (YOLO predicts all classes in a single forward pass per image)._

| Class Code | Mean Image Time (ms) | Images (n) |
|--- |--- |--- |
| BROST | 70.2 | 39 |
| STEME | 47.0 | 37 |
| URTUR | 44.9 | 31 |

## Setup Configuration Context
- **Model Identifier:** `yolo11s_ep150_b8_lr0.001`
- **Eval Platform:** `desktop`
- **Device Used:** `0`
- **Data Type:** `float32`
