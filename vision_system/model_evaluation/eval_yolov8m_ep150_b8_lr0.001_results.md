# Evaluation Report: YOLO (yolov8m_ep150_b8_lr0.001)

## Core Metrics Summary
- **Precision:** 0.8646
- **Recall:** 0.8834
- **F1 Score:** 0.8739
- **Mean IoU (Matched):** 0.8313

## Bounding Box Breakdown
- **Total Ground Truth Boxes:** 2162
- **Total Predicted Boxes:** 2209
- **True Positives (TP):** 1910
- **False Positives (FP):** 299
- **False Negatives (FN):** 252

## Per-Class Metrics Table
| Class Code | TP | FP | FN | Precision | Recall | F1 Score |
|--- |--- |--- |--- |--- |--- |--- |
| BROST | 250 | 59 | 47 | 0.8091 | 0.8418 | 0.8251 |
| STEME | 806 | 150 | 145 | 0.8431 | 0.8475 | 0.8453 |
| URTUR | 854 | 90 | 60 | 0.9047 | 0.9344 | 0.9193 |

## Setup Configuration Context
- **Model Identifier:** `yolov8m_ep150_b8_lr0.001`
- **Device Used:** `cpu`
- **Data Type:** `float32`
