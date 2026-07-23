# Evaluation Report: YOLO (yolov10m_ep150_b8_lr0.001)

## Core Metrics Summary
- **Precision:** 0.8547
- **Recall:** 0.8649
- **F1 Score:** 0.8598
- **Mean IoU (Matched):** 0.8255

## Bounding Box Breakdown
- **Total Ground Truth Boxes:** 2162
- **Total Predicted Boxes:** 2188
- **True Positives (TP):** 1870
- **False Positives (FP):** 318
- **False Negatives (FN):** 292

## Per-Class Metrics Table
| Class Code | TP | FP | FN | Precision | Recall | F1 Score |
|--- |--- |--- |--- |--- |--- |--- |
| BROST | 242 | 66 | 55 | 0.7857 | 0.8148 | 0.8000 |
| STEME | 784 | 160 | 167 | 0.8305 | 0.8244 | 0.8274 |
| URTUR | 844 | 92 | 70 | 0.9017 | 0.9234 | 0.9124 |

## Inference Timing — Overall
- **Mean:** 63.6 ms  (15.72 FPS)
- **Median:** 21.8 ms
- **Std Dev:** 92.5 ms
- **Mean (excl. first image / warm-up):** 56.1 ms (17.83 FPS)

## Inference Timing — By Class Present in Image
_Note: this groups images by which ground-truth classes they contain; it is the mean total inference time for images containing that class, not a per-detection or per-class model cost (YOLO predicts all classes in a single forward pass per image)._

| Class Code | Mean Image Time (ms) | Images (n) |
|--- |--- |--- |
| BROST | 88.2 | 39 |
| STEME | 49.3 | 37 |
| URTUR | 47.0 | 31 |

## Setup Configuration Context
- **Model Identifier:** `yolov10m_ep150_b8_lr0.001`
- **Eval Platform:** `desktop`
- **Device Used:** `0`
- **Data Type:** `float32`
