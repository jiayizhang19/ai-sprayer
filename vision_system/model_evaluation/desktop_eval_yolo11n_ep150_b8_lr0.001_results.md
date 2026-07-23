# Evaluation Report: YOLO (yolo11n_ep150_b8_lr0.001)

## Core Metrics Summary
- **Precision:** 0.8252
- **Recall:** 0.8497
- **F1 Score:** 0.8373
- **Mean IoU (Matched):** 0.8384

## Bounding Box Breakdown
- **Total Ground Truth Boxes:** 2162
- **Total Predicted Boxes:** 2226
- **True Positives (TP):** 1837
- **False Positives (FP):** 389
- **False Negatives (FN):** 325

## Per-Class Metrics Table
| Class Code | TP | FP | FN | Precision | Recall | F1 Score |
|--- |--- |--- |--- |--- |--- |--- |
| BROST | 236 | 50 | 61 | 0.8252 | 0.7946 | 0.8096 |
| STEME | 781 | 247 | 170 | 0.7597 | 0.8212 | 0.7893 |
| URTUR | 820 | 92 | 94 | 0.8991 | 0.8972 | 0.8981 |

## Inference Timing — Overall
- **Mean:** 55.9 ms  (17.90 FPS)
- **Median:** 23.1 ms
- **Std Dev:** 63.6 ms
- **Mean (excl. first image / warm-up):** 50.9 ms (19.66 FPS)

## Inference Timing — By Class Present in Image
_Note: this groups images by which ground-truth classes they contain; it is the mean total inference time for images containing that class, not a per-detection or per-class model cost (YOLO predicts all classes in a single forward pass per image)._

| Class Code | Mean Image Time (ms) | Images (n) |
|--- |--- |--- |
| BROST | 61.6 | 39 |
| STEME | 52.5 | 37 |
| URTUR | 51.2 | 31 |

## Setup Configuration Context
- **Model Identifier:** `yolo11n_ep150_b8_lr0.001`
- **Eval Platform:** `desktop`
- **Device Used:** `0`
- **Data Type:** `float32`
