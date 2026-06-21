# Weed Detection — Parameter Tuning Guide

`weed_detection_step1.py` runs zero-shot weed detection using LocateAnything-3B.
All settings live in **`config/config.yaml`** — edit that file to change
behaviour. No need to touch the Python script.

## Project layout

```
AI-Sprayer/
├── README.md                          <- project usage guide
├── requirements.txt                   <- Python dependency list
├── yolov8n.pt                         <- base YOLO checkpoint
└── vision_system/
  ├── config/
  │   ├── config.yaml                <- main runtime settings, including model selection and evaluation settings
  │   ├── config_loader.py           <- loads and maps config
  │   ├── data.yaml                  <- dataset split and classes
  │   └── train_config.yaml          <- YOLO training settings, including hyperparameters and optimizer settings for training purposes only
  ├── src/
  │   ├── weed_detection.py          <- run weed detection inference, main code for class and centroid detection, parameters in `config/config.yaml`
  │   ├── detection_evaluation.py    <- evaluate model performance on test set, parameters in `config/config.yaml`
  │   ├── train_yolo.py              <- fine-tune YOLO weights, parameters in `config/train_config.yaml`
  │   ├── ground_truth_verification.py <- visualize GT label quality for one time check
  │   ├── analyze_class_distribution.py <- dataset distribution analysis
  │   └── generate_run_report.py     <- summarize run metrics (one time runner for historical runs)
  ├── images/                        <- input images including training, validation, and test sets
  ├── labels/                        <- YOLO label annotations
  ├── runs/                          <- training run artifacts, including weights and metrics
  ├── model_evaluation/              <- evaluate model performance on test set
  ├── weed_detections/               <- annotated detection images
  └── weed_detections.json           <- current detection export to ros2
```
## Detection Models

### YOLOv8

### Locate Anything Parameters (in `config/config.yaml`)

| Parameter | Section | Default | Adjust when... |
|---|---|---|---|
| `device` | `model` | `"cpu"` | You have an NVIDIA GPU available → set to `"cuda"` |
| `dtype` | `model` | `"float32"` | On GPU → set to `"float16"` for faster inference |
| `input_dir` | `paths` | `weed_images` | Point at your image folder |
| `output_dir` | `paths` | `weed_detections` | Change where annotated images are saved |
| `results_file` | `paths` | `detections.json` | Change where detection results are saved |
| `prompts` | `detection` | `"Locate all weeds in this image."` | Results are noisy or missing certain weed types → try a more specific prompt |
| `min_box_area_fraction` | `filtering` | `0.005` | Small junk boxes still appear → **increase**. Real small/young weeds get dropped → **decrease** |
| `containment_threshold` | `filtering` | `0.9` | Junk boxes overlapping real detections aren't removed → **decrease** (e.g. 0.8). Adjacent real weeds get merged into one box → **increase** (closer to 1.0) |
| `repetition_penalty` | `generation` | `1.3` | Model still loops/repeats boxes → **increase** (e.g. 1.5). Model misses legitimately repeated detections → **decrease** |
| `no_repeat_ngram_size` | `generation` | `8` | Loops persist → **decrease** (e.g. 4–6). Output cut off too early → **increase** |
| `max_new_tokens` | `generation` | `512` | Many weeds per image and output gets cut off → **increase** |

### Quick checks for Locate Anything output

- **0 detections** → check `raw_model_output` in `detections.json`; if boxes
  are present but `weed_count` is 0, lower `min_box_area_fraction`.
- **Too many boxes** → lower `containment_threshold` and/or raise
  `min_box_area_fraction`.
- **Slow / still looping** → raise `repetition_penalty` or lower
  `no_repeat_ngram_size`.