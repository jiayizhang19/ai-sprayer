# Evaluation Report: YOLO (yolo11m_ep150_b8_lr0.001)

## Core Metrics Summary
- **Precision:** 0.8672
- **Recall:** 0.8760
- **F1 Score:** 0.8716
- **Mean IoU (Matched):** 0.8240

## Bounding Box Breakdown
- **Total Ground Truth Boxes:** 2162
- **Total Predicted Boxes:** 2184
- **True Positives (TP):** 1894
- **False Positives (FP):** 290
- **False Negatives (FN):** 268

## Per-Class Metrics Table
| Class Code | TP | FP | FN | Precision | Recall | F1 Score |
|--- |--- |--- |--- |--- |--- |--- |
| STEME | 800 | 149 | 151 | 0.8430 | 0.8412 | 0.8421 |
| URTUR | 847 | 82 | 67 | 0.9117 | 0.9267 | 0.9192 |
| BROST | 247 | 59 | 50 | 0.8072 | 0.8316 | 0.8192 |

## Inference Timing — Overall
- **Mean:** 139.0 ms  (7.19 FPS)
- **Median:** 66.6 ms
- **Std Dev:** 230.4 ms
- **Mean (excl. first image / warm-up):** 117.8 ms (8.49 FPS)

## Inference Timing — By Class Present in Image
_Note: this groups images by which ground-truth classes they contain; it is the mean total inference time for images containing that class, not a per-detection or per-class model cost (YOLO predicts all classes in a single forward pass per image)._

| Class Code | Mean Image Time (ms) | Images (n) |
|--- |--- |--- |
| STEME | 181.9 | 37 |
| URTUR | 123.2 | 31 |
| BROST | 109.8 | 39 |

## Setup Configuration Context
- **Model Identifier:** `yolo11m_ep150_b8_lr0.001`
- **Eval Platform:** `Orin`
- **Device Used:** `0`
- **Data Type:** `float32`
