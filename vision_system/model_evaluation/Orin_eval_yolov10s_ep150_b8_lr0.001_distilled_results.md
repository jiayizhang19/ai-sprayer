# Evaluation Report: YOLO (yolov10s_ep150_b8_lr0.001_distilled)

## Core Metrics Summary
- **Precision:** 0.8660
- **Recall:** 0.8575
- **F1 Score:** 0.8617
- **Mean IoU (Matched):** 0.8280

## Bounding Box Breakdown
- **Total Ground Truth Boxes:** 2162
- **Total Predicted Boxes:** 2141
- **True Positives (TP):** 1854
- **False Positives (FP):** 287
- **False Negatives (FN):** 308

## Per-Class Metrics Table
| Class Code | TP | FP | FN | Precision | Recall | F1 Score |
|--- |--- |--- |--- |--- |--- |--- |
| URTUR | 839 | 94 | 75 | 0.8992 | 0.9179 | 0.9085 |
| STEME | 782 | 164 | 169 | 0.8266 | 0.8223 | 0.8245 |
| BROST | 233 | 29 | 64 | 0.8893 | 0.7845 | 0.8336 |

## Inference Timing — Overall
### End-to-End (wall-clock, includes pre/post-process + Python overhead)
- **Mean:** 103.4 ms  (9.67 FPS)
- **Median:** 44.6 ms
- **Std Dev:** 129.0 ms
- **Mean (excl. first image / warm-up):** 92.4 ms (10.82 FPS)

### Pure Neural-Network Inference (Ultralytics `result.speed['inference']`)
- **Mean:** 31.7 ms  (31.55 FPS)
- **Median:** 29.3 ms
- **Std Dev:** 12.6 ms
- **Mean (excl. first image / warm-up):** 30.6 ms (32.67 FPS)

## Inference Timing — By Class Present in Image
_Note: this groups images by which ground-truth classes they contain; it is the mean total inference time for images containing that class, not a per-detection or per-class model cost (YOLO predicts all classes in a single forward pass per image)._

| Class Code | Mean Image Time (ms) | Images (n) |
|--- |--- |--- |
| STEME | 127.4 | 37 |
| URTUR | 95.4 | 31 |
| BROST | 85.6 | 39 |

## Setup Configuration Context
- **Model Identifier:** `yolov10s_ep150_b8_lr0.001_distilled`
- **Eval Platform:** `Orin`
- **Device Used:** `0`
- **Data Type:** `float16`
