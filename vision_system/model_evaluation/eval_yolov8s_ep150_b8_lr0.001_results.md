# Evaluation Report: YOLO (yolov8s_ep150_b8_lr0.001)

## Core Metrics Summary
- **Precision:** 0.8652
- **Recall:** 0.8756
- **F1 Score:** 0.8703
- **Mean IoU (Matched):** 0.8373

## Bounding Box Breakdown
- **Total Ground Truth Boxes:** 2162
- **Total Predicted Boxes:** 2188
- **True Positives (TP):** 1893
- **False Positives (FP):** 295
- **False Negatives (FN):** 269

## Per-Class Metrics Table
| Class Code | TP | FP | FN | Precision | Recall | F1 Score |
|--- |--- |--- |--- |--- |--- |--- |
| BROST | 240 | 47 | 57 | 0.8362 | 0.8081 | 0.8219 |
| STEME | 802 | 157 | 149 | 0.8363 | 0.8433 | 0.8398 |
| URTUR | 851 | 91 | 63 | 0.9034 | 0.9311 | 0.9170 |

## Setup Configuration Context
- **Model Identifier:** `yolov8s_ep150_b8_lr0.001`
- **Device Used:** `cpu`
- **Data Type:** `float32`
