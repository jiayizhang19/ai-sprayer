# Evaluation Report: YOLO (yolo11m_ep150_b8_lr0.001)

## Core Metrics Summary
- **Precision:** 0.8676
- **Recall:** 0.8760
- **F1 Score:** 0.8718
- **Mean IoU (Matched):** 0.8240

## Bounding Box Breakdown
- **Total Ground Truth Boxes:** 2162
- **Total Predicted Boxes:** 2183
- **True Positives (TP):** 1894
- **False Positives (FP):** 289
- **False Negatives (FN):** 268

## Per-Class Metrics Table
| Class Code | TP | FP | FN | Precision | Recall | F1 Score |
|--- |--- |--- |--- |--- |--- |--- |
| BROST | 247 | 59 | 50 | 0.8072 | 0.8316 | 0.8192 |
| STEME | 800 | 148 | 151 | 0.8439 | 0.8412 | 0.8425 |
| URTUR | 847 | 82 | 67 | 0.9117 | 0.9267 | 0.9192 |

## Inference Timing — Overall
### End-to-End (wall-clock, includes pre/post-process + Python overhead)
- **Mean:** 54.9 ms  (18.22 FPS)
- **Median:** 24.7 ms
- **Std Dev:** 64.5 ms
- **Mean (excl. first image / warm-up):** 49.4 ms (20.22 FPS)

### Pure Neural-Network Inference (Ultralytics `result.speed['inference']`)
- **Mean:** 18.9 ms  (52.87 FPS)
- **Median:** 16.1 ms
- **Std Dev:** 8.1 ms
- **Mean (excl. first image / warm-up):** 18.9 ms (52.83 FPS)

## Inference Timing — By Class Present in Image
_Note: this groups images by which ground-truth classes they contain; it is the mean total inference time for images containing that class, not a per-detection or per-class model cost (YOLO predicts all classes in a single forward pass per image)._

| Class Code | Mean Image Time (ms) | Images (n) |
|--- |--- |--- |
| BROST | 66.1 | 39 |
| STEME | 48.4 | 37 |
| URTUR | 47.4 | 31 |

## Setup Configuration Context
- **Model Identifier:** `yolo11m_ep150_b8_lr0.001`
- **Eval Platform:** `desktop`
- **Device Used:** `0`
- **Data Type:** `float32`
