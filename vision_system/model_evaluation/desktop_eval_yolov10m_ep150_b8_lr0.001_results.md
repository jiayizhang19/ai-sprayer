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
### End-to-End (wall-clock, includes pre/post-process + Python overhead)
- **Mean:** 58.0 ms  (17.23 FPS)
- **Median:** 21.7 ms
- **Std Dev:** 65.0 ms
- **Mean (excl. first image / warm-up):** 53.3 ms (18.77 FPS)

### Pure Neural-Network Inference (Ultralytics `result.speed['inference']`)
- **Mean:** 20.6 ms  (48.66 FPS)
- **Median:** 14.9 ms
- **Std Dev:** 18.3 ms
- **Mean (excl. first image / warm-up):** 20.6 ms (48.56 FPS)

## Inference Timing — By Class Present in Image
_Note: this groups images by which ground-truth classes they contain; it is the mean total inference time for images containing that class, not a per-detection or per-class model cost (YOLO predicts all classes in a single forward pass per image)._

| Class Code | Mean Image Time (ms) | Images (n) |
|--- |--- |--- |
| BROST | 74.2 | 39 |
| STEME | 49.0 | 37 |
| URTUR | 47.3 | 31 |

## Setup Configuration Context
- **Model Identifier:** `yolov10m_ep150_b8_lr0.001`
- **Eval Platform:** `desktop`
- **Device Used:** `0`
- **Data Type:** `float32`
