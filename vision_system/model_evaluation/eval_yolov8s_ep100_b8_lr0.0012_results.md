# Evaluation Report: YOLO (yolov8s_ep100_b8_lr0.0012)

## Core Metrics Summary
- **Precision:** 0.8417
- **Recall:** 0.8511
- **F1 Score:** 0.8464
- **Mean IoU (Matched):** 0.8261

## Bounding Box Breakdown
- **Total Ground Truth Boxes:** 2162
- **Total Predicted Boxes:** 2186
- **True Positives (TP):** 1840
- **False Positives (FP):** 346
- **False Negatives (FN):** 322

## Per-Class Metrics Table
| Class Code | TP | FP | FN | Precision | Recall | F1 Score |
|--- |--- |--- |--- |--- |--- |--- |
| BROST | 243 | 42 | 54 | 0.8526 | 0.8182 | 0.8351 |
| STEME | 775 | 241 | 176 | 0.7628 | 0.8149 | 0.7880 |
| URTUR | 822 | 63 | 92 | 0.9288 | 0.8993 | 0.9138 |

## Setup Configuration Context
- **Model Identifier:** `yolov8s_ep100_b8_lr0.0012`
- **Device Used:** `cpu`
- **Data Type:** `float32`
