# Evaluation Report: YOLO (yolo26n_ep150_b8_lr0.001)

## Core Metrics Summary
- **Precision:** 0.8535
- **Recall:** 0.8247
- **F1 Score:** 0.8389
- **Mean IoU (Matched):** 0.8405

## Bounding Box Breakdown
- **Total Ground Truth Boxes:** 2162
- **Total Predicted Boxes:** 2089
- **True Positives (TP):** 1783
- **False Positives (FP):** 306
- **False Negatives (FN):** 379

## Per-Class Metrics Table
| Class Code | TP | FP | FN | Precision | Recall | F1 Score |
|--- |--- |--- |--- |--- |--- |--- |
| URTUR | 808 | 87 | 106 | 0.9028 | 0.8840 | 0.8933 |
| STEME | 742 | 182 | 209 | 0.8030 | 0.7802 | 0.7915 |
| BROST | 233 | 37 | 64 | 0.8630 | 0.7845 | 0.8219 |

## Inference Timing — Overall
- **Mean:** 119.5 ms  (8.37 FPS)
- **Median:** 47.3 ms
- **Std Dev:** 290.7 ms
- **Mean (excl. first image / warm-up):** 92.1 ms (10.86 FPS)

## Inference Timing — By Class Present in Image
_Note: this groups images by which ground-truth classes they contain; it is the mean total inference time for images containing that class, not a per-detection or per-class model cost (YOLO predicts all classes in a single forward pass per image)._

| Class Code | Mean Image Time (ms) | Images (n) |
|--- |--- |--- |
| STEME | 174.5 | 37 |
| URTUR | 96.0 | 31 |
| BROST | 84.7 | 39 |

## Setup Configuration Context
- **Model Identifier:** `yolo26n_ep150_b8_lr0.001`
- **Eval Platform:** `Orin`
- **Device Used:** `0`
- **Data Type:** `float32`
