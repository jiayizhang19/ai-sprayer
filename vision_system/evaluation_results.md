# LocateAnything vs Ground Truth Evaluation

## Experiment Config

- Model ID: nvidia/LocateAnything-3B
- Device: cpu
- Dtype: float32
- Prompt used: Locate all weeds in this image.
- min_box_area_fraction: 0.005
- containment_threshold: 0.9
- conf_threshold: 0.0
- max_new_tokens: 512
- repetition_penalty: 1.3
- no_repeat_ngram_size: 8

- IoU threshold: 0.5
- Total ground truth boxes: 19
- Total predictions: 15
- True positives (TP): 12
- False positives (FP): 3
- False negatives (FN): 7
- Precision: 0.8000
- Recall: 0.6316
- F1 score: 0.7059
- Mean IoU (matched): 0.7854
- Total matched boxes: 12

## Per-image results

| Image | GT | Pred | TP | FP | FN | Precision | Recall | F1 | Mean IoU |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| capture_001_Brome_png.rf.4528194d80cc6998144b09c8d958a20c.jpg | 3 | 2 | 2 | 0 | 1 | 1.0000 | 0.6667 | 0.8000 | 0.7818 |
| capture_001_Brome_png.rf.c44c6df6ad78b6aab5b1b9f5ad777438.jpg | 3 | 2 | 2 | 0 | 1 | 1.0000 | 0.6667 | 0.8000 | 0.7534 |
| capture_001_Brome_png.rf.c803c3d989640e165b81a1046db6fd84.jpg | 3 | 3 | 2 | 1 | 1 | 0.6667 | 0.6667 | 0.6667 | 0.8355 |
| capture_002_Brome_png.rf.404cb3bc2adbc025d9d84ade5f47c835.jpg | 5 | 4 | 2 | 2 | 3 | 0.5000 | 0.4000 | 0.4444 | 0.7645 |
| capture_002_Brome_png.rf.94f9812f9f9cbb79d607c8aa1ff4c924.jpg | 5 | 4 | 4 | 0 | 1 | 1.0000 | 0.8000 | 0.8889 | 0.7887 |

## Notes
- Ground truth boxes are loaded from YOLO `.txt` files in `labels/`.
- Predictions are loaded from `detections.json` produced by the detection script.
- IoU matching uses a greedy one-to-one assignment at the threshold above.
