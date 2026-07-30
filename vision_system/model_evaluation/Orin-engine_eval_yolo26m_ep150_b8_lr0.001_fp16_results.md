# Evaluation Report: YOLO (yolo26m_ep150_b8_lr0.001_fp16)

## Core Metrics Summary
- **Precision:** 0.8656
- **Recall:** 0.8700
- **F1 Score:** 0.8678
- **Mean IoU (Matched):** 0.8428

## Bounding Box Breakdown
- **Total Ground Truth Boxes:** 2162
- **Total Predicted Boxes:** 2173
- **True Positives (TP):** 1881
- **False Positives (FP):** 292
- **False Negatives (FN):** 281

## Per-Class Metrics Table
| Class Code | TP | FP | FN | Precision | Recall | F1 Score |
|--- |--- |--- |--- |--- |--- |--- |
| URTUR | 842 | 105 | 72 | 0.8891 | 0.9212 | 0.9049 |
| STEME | 789 | 141 | 162 | 0.8484 | 0.8297 | 0.8389 |
| BROST | 250 | 46 | 47 | 0.8446 | 0.8418 | 0.8432 |

## Inference Timing — Overall
### End-to-End (wall-clock, includes pre/post-process + Python overhead)
- **Mean:** 87.2 ms  (11.47 FPS)
- **Median:** 35.2 ms
- **Std Dev:** 94.8 ms
- **Mean (excl. first image / warm-up):** 80.4 ms (12.43 FPS)

### Pure Neural-Network Inference (Ultralytics `result.speed['inference']`)
- **Mean:** 18.6 ms  (53.77 FPS)
- **Median:** 17.5 ms
- **Std Dev:** 8.0 ms
- **Mean (excl. first image / warm-up):** 18.6 ms (53.76 FPS)

## Inference Timing — By Class Present in Image
_Note: this groups images by which ground-truth classes they contain; it is the mean total inference time for images containing that class, not a per-detection or per-class model cost (YOLO predicts all classes in a single forward pass per image)._

| Class Code | Mean Image Time (ms) | Images (n) |
|--- |--- |--- |
| STEME | 102.3 | 37 |
| URTUR | 83.3 | 31 |
| BROST | 74.4 | 39 |

## Setup Configuration Context
- **Model Identifier:** `yolo26m_ep150_b8_lr0.001_fp16`
- **Eval Platform:** `Orin-engine`
- **Device Used:** `0`
- **Data Type:** `float32`
