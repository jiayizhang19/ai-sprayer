# Evaluation Report: YOLO (yolo26s_ep150_b32_lr0.001)

## Core Metrics Summary
- **Precision:** 0.8512
- **Recall:** 0.8626
- **F1 Score:** 0.8569
- **Mean IoU (Matched):** 0.8399

## Bounding Box Breakdown
- **Total Ground Truth Boxes:** 2162
- **Total Predicted Boxes:** 2191
- **True Positives (TP):** 1865
- **False Positives (FP):** 326
- **False Negatives (FN):** 297

## Per-Class Metrics Table
| Class Code | TP | FP | FN | Precision | Recall | F1 Score |
|--- |--- |--- |--- |--- |--- |--- |
| BROST | 246 | 64 | 51 | 0.7935 | 0.8283 | 0.8105 |
| STEME | 770 | 161 | 181 | 0.8271 | 0.8097 | 0.8183 |
| URTUR | 849 | 101 | 65 | 0.8937 | 0.9289 | 0.9109 |

## Inference Timing — Overall
- **Mean:** 57.5 ms  (17.40 FPS)
- **Median:** 14.7 ms
- **Std Dev:** 94.9 ms
- **Mean (excl. first image / warm-up):** 49.6 ms (20.17 FPS)

## Inference Timing — By Class Present in Image
_Note: this groups images by which ground-truth classes they contain; it is the mean total inference time for images containing that class, not a per-detection or per-class model cost (YOLO predicts all classes in a single forward pass per image)._

| Class Code | Mean Image Time (ms) | Images (n) |
|--- |--- |--- |
| BROST | 84.1 | 39 |
| STEME | 41.4 | 37 |
| URTUR | 41.4 | 31 |

## Setup Configuration Context
- **Model Identifier:** `yolo26s_ep150_b32_lr0.001`
- **Eval Platform:** `desktop`
- **Device Used:** `0`
- **Data Type:** `float32`
