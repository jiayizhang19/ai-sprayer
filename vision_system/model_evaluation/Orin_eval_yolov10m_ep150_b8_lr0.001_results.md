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
### End-to-End (wall-clock, includes pre/post-process + Python overhead)
- **Mean:** 129.6 ms  (7.72 FPS)
- **Median:** 70.1 ms
- **Std Dev:** 132.8 ms
- **Mean (excl. first image / warm-up):** 118.3 ms (8.45 FPS)

### Pure Neural-Network Inference (Ultralytics `result.speed['inference']`)
- **Mean:** 56.1 ms  (17.82 FPS)
- **Median:** 53.5 ms
- **Std Dev:** 11.7 ms
- **Mean (excl. first image / warm-up):** 55.2 ms (18.10 FPS)

## Inference Timing — By Class Present in Image
_Note: this groups images by which ground-truth classes they contain; it is the mean total inference time for images containing that class, not a per-detection or per-class model cost (YOLO predicts all classes in a single forward pass per image)._

| Class Code | Mean Image Time (ms) | Images (n) |
|--- |--- |--- |
| STEME | 154.9 | 37 |
| URTUR | 119.8 | 31 |
| BROST | 111.7 | 39 |

## Setup Configuration Context
- **Model Identifier:** `yolov10m_ep150_b8_lr0.001`
- **Eval Platform:** `Orin`
- **Device Used:** `0`
- **Data Type:** `float32`
