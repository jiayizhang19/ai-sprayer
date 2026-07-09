# Evaluation Report: YOLO (yolov8n_ep150_b8_lr0.001)

## Core Metrics Summary
- **Precision:** 0.8254
- **Recall:** 0.8612
- **F1 Score:** 0.8429
- **Mean IoU (Matched):** 0.8338

## Bounding Box Breakdown
- **Total Ground Truth Boxes:** 2162
- **Total Predicted Boxes:** 2256
- **True Positives (TP):** 1862
- **False Positives (FP):** 394
- **False Negatives (FN):** 300

## Per-Class Metrics Table
| Class Code | TP | FP | FN | Precision | Recall | F1 Score |
|--- |--- |--- |--- |--- |--- |--- |
| BROST | 239 | 66 | 58 | 0.7836 | 0.8047 | 0.7940 |
| STEME | 791 | 257 | 160 | 0.7548 | 0.8318 | 0.7914 |
| URTUR | 832 | 71 | 82 | 0.9214 | 0.9103 | 0.9158 |

## Setup Configuration Context
- **Model Identifier:** `yolov8n_ep150_b8_lr0.001`
- **Device Used:** `cpu`
- **Data Type:** `float32`
