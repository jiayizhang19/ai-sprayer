# Evaluation Report: YOLO (yolov8n_ep30_b8_lr0.0004)

## Core Metrics Summary
- **Precision:** 0.7709
- **Recall:** 0.7562
- **F1 Score:** 0.7635
- **Mean IoU (Matched):** 0.8047

## Bounding Box Breakdown
- **Total Ground Truth Boxes:** 2162
- **Total Predicted Boxes:** 2121
- **True Positives (TP):** 1635
- **False Positives (FP):** 486
- **False Negatives (FN):** 527

## Per-Class Metrics Table
| Class Code | TP | FP | FN | Precision | Recall | F1 Score |
|--- |--- |--- |--- |--- |--- |--- |
| BROST | 219 | 52 | 78 | 0.8081 | 0.7374 | 0.7711 |
| STEME | 674 | 260 | 277 | 0.7216 | 0.7087 | 0.7151 |
| URTUR | 742 | 174 | 172 | 0.8100 | 0.8118 | 0.8109 |

## Setup Configuration Context
- **Model Identifier:** `yolov8n_ep30_b8_lr0.0004`
- **Device Used:** `cpu`
- **Data Type:** `float32`
