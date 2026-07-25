# Evaluation Report: YOLO (yolov10m_ep150_b8_lr0.001)

## Core Metrics Summary
- **Precision:** 0.8542
- **Recall:** 0.8645
- **F1 Score:** 0.8593
- **Mean IoU (Matched):** 0.8257

## Bounding Box Breakdown
- **Total Ground Truth Boxes:** 2162
- **Total Predicted Boxes:** 2188
- **True Positives (TP):** 1869
- **False Positives (FP):** 319
- **False Negatives (FN):** 293

## Per-Class Metrics Table
| Class Code | TP | FP | FN | Precision | Recall | F1 Score |
|--- |--- |--- |--- |--- |--- |--- |
| STEME | 783 | 161 | 168 | 0.8294 | 0.8233 | 0.8264 |
| URTUR | 844 | 92 | 70 | 0.9017 | 0.9234 | 0.9124 |
| BROST | 242 | 66 | 55 | 0.7857 | 0.8148 | 0.8000 |

## Inference Timing — Overall
- **Mean:** 125.7 ms  (7.95 FPS)
- **Median:** 63.4 ms
- **Std Dev:** 116.5 ms
- **Mean (excl. first image / warm-up):** 116.7 ms (8.57 FPS)

## Inference Timing — By Class Present in Image
_Note: this groups images by which ground-truth classes they contain; it is the mean total inference time for images containing that class, not a per-detection or per-class model cost (YOLO predicts all classes in a single forward pass per image)._

| Class Code | Mean Image Time (ms) | Images (n) |
|--- |--- |--- |
| STEME | 143.4 | 37 |
| URTUR | 117.8 | 31 |
| BROST | 112.9 | 39 |

## Setup Configuration Context
- **Model Identifier:** `yolov10m_ep150_b8_lr0.001`
- **Eval Platform:** `Orin`
- **Device Used:** `0`
- **Data Type:** `float32`
