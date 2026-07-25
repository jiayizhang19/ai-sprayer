# Evaluation Report: YOLO (yolov10n_ep150_b8_lr0.001)

## Core Metrics Summary
- **Precision:** 0.8239
- **Recall:** 0.8330
- **F1 Score:** 0.8284
- **Mean IoU (Matched):** 0.8364

## Bounding Box Breakdown
- **Total Ground Truth Boxes:** 2162
- **Total Predicted Boxes:** 2186
- **True Positives (TP):** 1801
- **False Positives (FP):** 385
- **False Negatives (FN):** 361

## Per-Class Metrics Table
| Class Code | TP | FP | FN | Precision | Recall | F1 Score |
|--- |--- |--- |--- |--- |--- |--- |
| STEME | 768 | 237 | 183 | 0.7642 | 0.8076 | 0.7853 |
| URTUR | 801 | 99 | 113 | 0.8900 | 0.8764 | 0.8831 |
| BROST | 232 | 49 | 65 | 0.8256 | 0.7811 | 0.8028 |

## Inference Timing — Overall
- **Mean:** 99.4 ms  (10.06 FPS)
- **Median:** 49.5 ms
- **Std Dev:** 101.3 ms
- **Mean (excl. first image / warm-up):** 91.4 ms (10.94 FPS)

## Inference Timing — By Class Present in Image
_Note: this groups images by which ground-truth classes they contain; it is the mean total inference time for images containing that class, not a per-detection or per-class model cost (YOLO predicts all classes in a single forward pass per image)._

| Class Code | Mean Image Time (ms) | Images (n) |
|--- |--- |--- |
| STEME | 117.4 | 37 |
| URTUR | 92.7 | 31 |
| BROST | 85.9 | 39 |

## Setup Configuration Context
- **Model Identifier:** `yolov10n_ep150_b8_lr0.001`
- **Eval Platform:** `Orin`
- **Device Used:** `0`
- **Data Type:** `float32`
