# Evaluation Report: YOLO (yolov10m_ep150_b8_lr0.001_fp16)

## Core Metrics Summary
- **Precision:** 0.8439
- **Recall:** 0.8501
- **F1 Score:** 0.8470
- **Mean IoU (Matched):** 0.8343

## Bounding Box Breakdown
- **Total Ground Truth Boxes:** 2162
- **Total Predicted Boxes:** 2178
- **True Positives (TP):** 1838
- **False Positives (FP):** 340
- **False Negatives (FN):** 324

## Per-Class Metrics Table
| Class Code | TP | FP | FN | Precision | Recall | F1 Score |
|--- |--- |--- |--- |--- |--- |--- |
| URTUR | 826 | 111 | 88 | 0.8815 | 0.9037 | 0.8925 |
| STEME | 774 | 181 | 177 | 0.8105 | 0.8139 | 0.8122 |
| BROST | 238 | 48 | 59 | 0.8322 | 0.8013 | 0.8165 |

## Inference Timing — Overall
### End-to-End (wall-clock, includes pre/post-process + Python overhead)
- **Mean:** 97.4 ms  (10.27 FPS)
- **Median:** 41.7 ms
- **Std Dev:** 95.2 ms
- **Mean (excl. first image / warm-up):** 91.0 ms (10.99 FPS)

### Pure Neural-Network Inference (Ultralytics `result.speed['inference']`)
- **Mean:** 28.0 ms  (35.66 FPS)
- **Median:** 25.9 ms
- **Std Dev:** 12.2 ms
- **Mean (excl. first image / warm-up):** 28.0 ms (35.70 FPS)

## Inference Timing — By Class Present in Image
_Note: this groups images by which ground-truth classes they contain; it is the mean total inference time for images containing that class, not a per-detection or per-class model cost (YOLO predicts all classes in a single forward pass per image)._

| Class Code | Mean Image Time (ms) | Images (n) |
|--- |--- |--- |
| STEME | 109.5 | 37 |
| URTUR | 95.7 | 31 |
| BROST | 85.7 | 39 |

## Setup Configuration Context
- **Model Identifier:** `yolov10m_ep150_b8_lr0.001_fp16`
- **Eval Platform:** `Orin-engine`
- **Device Used:** `0`
- **Data Type:** `float16`
