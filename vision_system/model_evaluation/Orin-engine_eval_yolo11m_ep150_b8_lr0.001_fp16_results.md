# Evaluation Report: YOLO (yolo11m_ep150_b8_lr0.001_fp16)

## Core Metrics Summary
- **Precision:** 0.8518
- **Recall:** 0.8612
- **F1 Score:** 0.8565
- **Mean IoU (Matched):** 0.8324

## Bounding Box Breakdown
- **Total Ground Truth Boxes:** 2162
- **Total Predicted Boxes:** 2186
- **True Positives (TP):** 1862
- **False Positives (FP):** 324
- **False Negatives (FN):** 300

## Per-Class Metrics Table
| Class Code | TP | FP | FN | Precision | Recall | F1 Score |
|--- |--- |--- |--- |--- |--- |--- |
| STEME | 779 | 176 | 172 | 0.8157 | 0.8191 | 0.8174 |
| URTUR | 840 | 97 | 74 | 0.8965 | 0.9190 | 0.9076 |
| BROST | 243 | 51 | 54 | 0.8265 | 0.8182 | 0.8223 |

## Inference Timing — Overall
### End-to-End (wall-clock, includes pre/post-process + Python overhead)
- **Mean:** 114.6 ms  (8.72 FPS)
- **Median:** 57.0 ms
- **Std Dev:** 219.3 ms
- **Mean (excl. first image / warm-up):** 94.1 ms (10.63 FPS)

### Pure Neural-Network Inference (Ultralytics `result.speed['inference']`)
- **Mean:** 26.7 ms  (37.43 FPS)
- **Median:** 27.2 ms
- **Std Dev:** 6.1 ms
- **Mean (excl. first image / warm-up):** 26.6 ms (37.61 FPS)

## Inference Timing — By Class Present in Image
_Note: this groups images by which ground-truth classes they contain; it is the mean total inference time for images containing that class, not a per-detection or per-class model cost (YOLO predicts all classes in a single forward pass per image)._

| Class Code | Mean Image Time (ms) | Images (n) |
|--- |--- |--- |
| STEME | 159.0 | 37 |
| URTUR | 96.6 | 31 |
| BROST | 86.2 | 39 |

## Setup Configuration Context
- **Model Identifier:** `yolo11m_ep150_b8_lr0.001_fp16`
- **Eval Platform:** `Orin-engine`
- **Device Used:** `0`
- **Data Type:** `float16`
