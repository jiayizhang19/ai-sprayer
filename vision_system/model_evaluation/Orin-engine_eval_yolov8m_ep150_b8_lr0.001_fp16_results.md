# Evaluation Report: YOLO (yolov8m_ep150_b8_lr0.001_fp16)

## Core Metrics Summary
- **Precision:** 0.8565
- **Recall:** 0.8696
- **F1 Score:** 0.8630
- **Mean IoU (Matched):** 0.8402

## Bounding Box Breakdown
- **Total Ground Truth Boxes:** 2162
- **Total Predicted Boxes:** 2195
- **True Positives (TP):** 1880
- **False Positives (FP):** 315
- **False Negatives (FN):** 282

## Per-Class Metrics Table
| Class Code | TP | FP | FN | Precision | Recall | F1 Score |
|--- |--- |--- |--- |--- |--- |--- |
| URTUR | 847 | 81 | 67 | 0.9127 | 0.9267 | 0.9197 |
| STEME | 788 | 187 | 163 | 0.8082 | 0.8286 | 0.8183 |
| BROST | 245 | 47 | 52 | 0.8390 | 0.8249 | 0.8319 |

## Inference Timing — Overall
### End-to-End (wall-clock, includes pre/post-process + Python overhead)
- **Mean:** 111.6 ms  (8.96 FPS)
- **Median:** 43.7 ms
- **Std Dev:** 224.3 ms
- **Mean (excl. first image / warm-up):** 90.8 ms (11.01 FPS)

### Pure Neural-Network Inference (Ultralytics `result.speed['inference']`)
- **Mean:** 24.3 ms  (41.21 FPS)
- **Median:** 21.5 ms
- **Std Dev:** 10.7 ms
- **Mean (excl. first image / warm-up):** 24.1 ms (41.45 FPS)

## Inference Timing — By Class Present in Image
_Note: this groups images by which ground-truth classes they contain; it is the mean total inference time for images containing that class, not a per-detection or per-class model cost (YOLO predicts all classes in a single forward pass per image)._

| Class Code | Mean Image Time (ms) | Images (n) |
|--- |--- |--- |
| STEME | 153.8 | 37 |
| URTUR | 92.1 | 31 |
| BROST | 85.5 | 39 |

## Setup Configuration Context
- **Model Identifier:** `yolov8m_ep150_b8_lr0.001_fp16`
- **Eval Platform:** `Orin-engine`
- **Device Used:** `0`
- **Data Type:** `float32`
