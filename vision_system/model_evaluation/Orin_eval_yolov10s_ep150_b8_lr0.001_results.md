# Evaluation Report: YOLO (yolov10s_ep150_b8_lr0.001)

## Core Metrics Summary
- **Precision:** 0.8411
- **Recall:** 0.8737
- **F1 Score:** 0.8571
- **Mean IoU (Matched):** 0.8265

## Bounding Box Breakdown
- **Total Ground Truth Boxes:** 2162
- **Total Predicted Boxes:** 2246
- **True Positives (TP):** 1889
- **False Positives (FP):** 357
- **False Negatives (FN):** 273

## Per-Class Metrics Table
| Class Code | TP | FP | FN | Precision | Recall | F1 Score |
|--- |--- |--- |--- |--- |--- |--- |
| STEME | 792 | 207 | 159 | 0.7928 | 0.8328 | 0.8123 |
| URTUR | 849 | 97 | 65 | 0.8975 | 0.9289 | 0.9129 |
| BROST | 248 | 53 | 49 | 0.8239 | 0.8350 | 0.8294 |

## Inference Timing — Overall
### End-to-End (wall-clock, includes pre/post-process + Python overhead)
- **Mean:** 108.1 ms  (9.25 FPS)
- **Median:** 51.8 ms
- **Std Dev:** 126.0 ms
- **Mean (excl. first image / warm-up):** 97.5 ms (10.25 FPS)

### Pure Neural-Network Inference (Ultralytics `result.speed['inference']`)
- **Mean:** 35.4 ms  (28.26 FPS)
- **Median:** 32.1 ms
- **Std Dev:** 13.3 ms
- **Mean (excl. first image / warm-up):** 34.4 ms (29.06 FPS)

## Inference Timing — By Class Present in Image
_Note: this groups images by which ground-truth classes they contain; it is the mean total inference time for images containing that class, not a per-detection or per-class model cost (YOLO predicts all classes in a single forward pass per image)._

| Class Code | Mean Image Time (ms) | Images (n) |
|--- |--- |--- |
| STEME | 128.2 | 37 |
| URTUR | 102.3 | 31 |
| BROST | 92.0 | 39 |

## Setup Configuration Context
- **Model Identifier:** `yolov10s_ep150_b8_lr0.001`
- **Eval Platform:** `Orin`
- **Device Used:** `0`
- **Data Type:** `float32`
