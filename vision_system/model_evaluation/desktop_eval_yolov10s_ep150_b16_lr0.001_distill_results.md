# Evaluation Report: YOLO (yolov10s_ep150_b16_lr0.001_distill)

## Core Metrics Summary
- **Precision:** 0.8660
- **Recall:** 0.8580
- **F1 Score:** 0.8620
- **Mean IoU (Matched):** 0.8280

## Bounding Box Breakdown
- **Total Ground Truth Boxes:** 2162
- **Total Predicted Boxes:** 2142
- **True Positives (TP):** 1855
- **False Positives (FP):** 287
- **False Negatives (FN):** 307

## Per-Class Metrics Table
| Class Code | TP | FP | FN | Precision | Recall | F1 Score |
|--- |--- |--- |--- |--- |--- |--- |
| BROST | 234 | 29 | 63 | 0.8897 | 0.7879 | 0.8357 |
| STEME | 782 | 164 | 169 | 0.8266 | 0.8223 | 0.8245 |
| URTUR | 839 | 94 | 75 | 0.8992 | 0.9179 | 0.9085 |

## Inference Timing — Overall
### End-to-End (wall-clock, includes pre/post-process + Python overhead)
- **Mean:** 62.5 ms  (16.01 FPS)
- **Median:** 14.5 ms
- **Std Dev:** 82.3 ms
- **Mean (excl. first image / warm-up):** 56.6 ms (17.67 FPS)

### Pure Neural-Network Inference (Ultralytics `result.speed['inference']`)
- **Mean:** 18.7 ms  (53.42 FPS)
- **Median:** 6.7 ms
- **Std Dev:** 24.4 ms
- **Mean (excl. first image / warm-up):** 18.8 ms (53.10 FPS)

## Inference Timing — By Class Present in Image
_Note: this groups images by which ground-truth classes they contain; it is the mean total inference time for images containing that class, not a per-detection or per-class model cost (YOLO predicts all classes in a single forward pass per image)._

| Class Code | Mean Image Time (ms) | Images (n) |
|--- |--- |--- |
| BROST | 83.6 | 39 |
| STEME | 50.9 | 37 |
| URTUR | 47.8 | 31 |

## Setup Configuration Context
- **Model Identifier:** `yolov10s_ep150_b16_lr0.001_distill`
- **Eval Platform:** `desktop`
- **Device Used:** `0`
- **Data Type:** `float32`
