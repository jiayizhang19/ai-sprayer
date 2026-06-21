# Evaluation Report: YOLO (yolov8n_ep40_b4_lr0.0002)

## Core Metrics Summary
- **Precision:** 0.7382
- **Recall:** 0.7225
- **F1 Score:** 0.7302
- **Mean IoU (Matched):** 0.7965

## Bounding Box Breakdown
- **Total Ground Truth Boxes:** 2162
- **Total Predicted Boxes:** 2116
- **True Positives (TP):** 1562
- **False Positives (FP):** 554
- **False Negatives (FN):** 600

## Per-Class Metrics Table
| Class Code | TP | FP | FN | Precision | Recall | F1 Score |
|--- |--- |--- |--- |--- |--- |--- |
| BROST | 218 | 60 | 79 | 0.7842 | 0.7340 | 0.7583 |
| STEME | 642 | 292 | 309 | 0.6874 | 0.6751 | 0.6812 |
| URTUR | 702 | 202 | 212 | 0.7765 | 0.7681 | 0.7723 |

## Setup Configuration Context
- **Model Identifier:** `yolov8n_ep40_b4_lr0.0002`
- **Device Used:** `cpu`
- **Data Type:** `float32`
