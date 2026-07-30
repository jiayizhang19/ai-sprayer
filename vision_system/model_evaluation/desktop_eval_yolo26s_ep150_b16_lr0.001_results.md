# Evaluation Report: YOLO (yolo26s_ep150_b16_lr0.001)

## Core Metrics Summary
- **Precision:** 0.8535
- **Recall:** 0.8705
- **F1 Score:** 0.8619
- **Mean IoU (Matched):** 0.8358

## Bounding Box Breakdown
- **Total Ground Truth Boxes:** 2162
- **Total Predicted Boxes:** 2205
- **True Positives (TP):** 1882
- **False Positives (FP):** 323
- **False Negatives (FN):** 280

## Per-Class Metrics Table
| Class Code | TP | FP | FN | Precision | Recall | F1 Score |
|--- |--- |--- |--- |--- |--- |--- |
| BROST | 246 | 36 | 51 | 0.8723 | 0.8283 | 0.8497 |
| STEME | 785 | 183 | 166 | 0.8110 | 0.8254 | 0.8181 |
| URTUR | 851 | 104 | 63 | 0.8911 | 0.9311 | 0.9106 |

## Inference Timing — Overall
### End-to-End (wall-clock, includes pre/post-process + Python overhead)
- **Mean:** 59.6 ms  (16.78 FPS)
- **Median:** 16.5 ms
- **Std Dev:** 82.6 ms
- **Mean (excl. first image / warm-up):** 53.4 ms (18.74 FPS)

### Pure Neural-Network Inference (Ultralytics `result.speed['inference']`)
- **Mean:** 19.2 ms  (52.10 FPS)
- **Median:** 8.0 ms
- **Std Dev:** 25.1 ms
- **Mean (excl. first image / warm-up):** 19.3 ms (51.82 FPS)

## Inference Timing — By Class Present in Image
_Note: this groups images by which ground-truth classes they contain; it is the mean total inference time for images containing that class, not a per-detection or per-class model cost (YOLO predicts all classes in a single forward pass per image)._

| Class Code | Mean Image Time (ms) | Images (n) |
|--- |--- |--- |
| BROST | 83.0 | 39 |
| STEME | 43.3 | 37 |
| URTUR | 47.8 | 31 |

## Setup Configuration Context
- **Model Identifier:** `yolo26s_ep150_b16_lr0.001`
- **Eval Platform:** `desktop`
- **Device Used:** `0`
- **Data Type:** `float32`
