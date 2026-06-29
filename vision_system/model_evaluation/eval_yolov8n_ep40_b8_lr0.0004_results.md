# Evaluation Report: YOLO (yolov8n_ep40_b8_lr0.0004)

## Core Metrics Summary
- **Precision:** 0.7780
- **Recall:** 0.7683
- **F1 Score:** 0.7731
- **Mean IoU (Matched):** 0.8120

## Bounding Box Breakdown
- **Total Ground Truth Boxes:** 2162
- **Total Predicted Boxes:** 2135
- **True Positives (TP):** 1661
- **False Positives (FP):** 474
- **False Negatives (FN):** 501

## Per-Class Metrics Table
| Class Code | TP | FP | FN | Precision | Recall | F1 Score |
|--- |--- |--- |--- |--- |--- |--- |
| BROST | 221 | 50 | 76 | 0.8155 | 0.7441 | 0.7782 |
| URTUR | 750 | 153 | 164 | 0.8306 | 0.8206 | 0.8255 |
| STEME | 690 | 271 | 261 | 0.7180 | 0.7256 | 0.7218 |

## Setup Configuration Context
- **Model Identifier:** `yolov8n_ep40_b8_lr0.0004`
- **Device Used:** `cpu`
- **Data Type:** `float32`
