# Evaluation Report: YOLO (yolov8n_ep150_b8_lr0.001)

## Core Metrics Summary
- **Precision:** 0.8254
- **Recall:** 0.8612
- **F1 Score:** 0.8429
- **Mean IoU (Matched):** 0.8338

## Bounding Box Breakdown
- **Total Ground Truth Boxes:** 2162
- **Total Predicted Boxes:** 2256
- **True Positives (TP):** 1862
- **False Positives (FP):** 394
- **False Negatives (FN):** 300

## Per-Class Metrics Table
| Class Code | TP | FP | FN | Precision | Recall | F1 Score |
|--- |--- |--- |--- |--- |--- |--- |
| BROST | 239 | 66 | 58 | 0.7836 | 0.8047 | 0.7940 |
| STEME | 791 | 257 | 160 | 0.7548 | 0.8318 | 0.7914 |
| URTUR | 832 | 71 | 82 | 0.9214 | 0.9103 | 0.9158 |

## Inference Timing — Overall
- **Mean:** 81.3 ms  (12.29 FPS)
- **Median:** 52.6 ms
- **Std Dev:** 37.6 ms
- **Mean (excl. first image / warm-up):** 80.1 ms (12.48 FPS)

## Inference Timing — By Class Present in Image
_Note: this groups images by which ground-truth classes they contain; it is the mean total inference time for images containing that class, not a per-detection or per-class model cost (YOLO predicts all classes in a single forward pass per image)._

| Class Code | Mean Image Time (ms) | Images (n) |
|--- |--- |--- |
| BROST | 83.1 | 39 |
| STEME | 80.3 | 37 |
| URTUR | 78.9 | 31 |

## Setup Configuration Context
- **Model Identifier:** `yolov8n_ep150_b8_lr0.001`
- **Eval Platform:** `laptop`
- **Device Used:** `cpu`
- **Data Type:** `float32`
