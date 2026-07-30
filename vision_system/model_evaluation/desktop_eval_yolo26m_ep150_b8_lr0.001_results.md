# Evaluation Report: YOLO (yolo26m_ep150_b8_lr0.001)

## Core Metrics Summary
- **Precision:** 0.8666
- **Recall:** 0.8802
- **F1 Score:** 0.8733
- **Mean IoU (Matched):** 0.8312

## Bounding Box Breakdown
- **Total Ground Truth Boxes:** 2162
- **Total Predicted Boxes:** 2196
- **True Positives (TP):** 1903
- **False Positives (FP):** 293
- **False Negatives (FN):** 259

## Per-Class Metrics Table
| Class Code | TP | FP | FN | Precision | Recall | F1 Score |
|--- |--- |--- |--- |--- |--- |--- |
| BROST | 252 | 49 | 45 | 0.8372 | 0.8485 | 0.8428 |
| STEME | 799 | 157 | 152 | 0.8358 | 0.8402 | 0.8380 |
| URTUR | 852 | 87 | 62 | 0.9073 | 0.9322 | 0.9196 |

## Inference Timing — Overall
### End-to-End (wall-clock, includes pre/post-process + Python overhead)
- **Mean:** 52.5 ms  (19.06 FPS)
- **Median:** 23.7 ms
- **Std Dev:** 59.9 ms
- **Mean (excl. first image / warm-up):** 47.4 ms (21.08 FPS)

### Pure Neural-Network Inference (Ultralytics `result.speed['inference']`)
- **Mean:** 18.1 ms  (55.10 FPS)
- **Median:** 16.6 ms
- **Std Dev:** 6.6 ms
- **Mean (excl. first image / warm-up):** 18.1 ms (55.11 FPS)

## Inference Timing — By Class Present in Image
_Note: this groups images by which ground-truth classes they contain; it is the mean total inference time for images containing that class, not a per-detection or per-class model cost (YOLO predicts all classes in a single forward pass per image)._

| Class Code | Mean Image Time (ms) | Images (n) |
|--- |--- |--- |
| BROST | 62.0 | 39 |
| STEME | 47.5 | 37 |
| URTUR | 45.8 | 31 |

## Setup Configuration Context
- **Model Identifier:** `yolo26m_ep150_b8_lr0.001`
- **Eval Platform:** `desktop`
- **Device Used:** `0`
- **Data Type:** `float32`
