# Evaluation Report: YOLO (yolov10s_ep150_b8_lr0.001)

## Core Metrics Summary
- **Precision:** 0.8414
- **Recall:** 0.8737
- **F1 Score:** 0.8573
- **Mean IoU (Matched):** 0.8265

## Bounding Box Breakdown
- **Total Ground Truth Boxes:** 2162
- **Total Predicted Boxes:** 2245
- **True Positives (TP):** 1889
- **False Positives (FP):** 356
- **False Negatives (FN):** 273

## Per-Class Metrics Table
| Class Code | TP | FP | FN | Precision | Recall | F1 Score |
|--- |--- |--- |--- |--- |--- |--- |
| BROST | 248 | 53 | 49 | 0.8239 | 0.8350 | 0.8294 |
| STEME | 792 | 206 | 159 | 0.7936 | 0.8328 | 0.8127 |
| URTUR | 849 | 97 | 65 | 0.8975 | 0.9289 | 0.9129 |

## Inference Timing — Overall
### End-to-End (wall-clock, includes pre/post-process + Python overhead)
- **Mean:** 39.3 ms  (25.47 FPS)
- **Median:** 13.4 ms
- **Std Dev:** 48.2 ms
- **Mean (excl. first image / warm-up):** 35.5 ms (28.20 FPS)

### Pure Neural-Network Inference (Ultralytics `result.speed['inference']`)
- **Mean:** 7.1 ms  (141.15 FPS)
- **Median:** 6.3 ms
- **Std Dev:** 5.6 ms
- **Mean (excl. first image / warm-up):** 7.1 ms (141.12 FPS)

## Inference Timing — By Class Present in Image
_Note: this groups images by which ground-truth classes they contain; it is the mean total inference time for images containing that class, not a per-detection or per-class model cost (YOLO predicts all classes in a single forward pass per image)._

| Class Code | Mean Image Time (ms) | Images (n) |
|--- |--- |--- |
| BROST | 43.2 | 39 |
| STEME | 37.9 | 37 |
| URTUR | 35.2 | 31 |

## Setup Configuration Context
- **Model Identifier:** `yolov10s_ep150_b8_lr0.001`
- **Eval Platform:** `desktop`
- **Device Used:** `0`
- **Data Type:** `float32`
