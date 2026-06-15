# Weed Detection — Parameter Tuning Guide

`weed_detection_step1.py` runs zero-shot weed detection using LocateAnything-3B.
All settings live in **`config/config.yaml`** — edit that file to change
behaviour. No need to touch the Python script.

## Project layout

```
vision_system/
├── config/
│   ├── config.yaml        <- edit this file
│   └── config_loader.py    (internal — no need to edit)
├── src/
│   └── weed_detection_step1.py
├── weed_images/             <- put your input images here
├── weed_detections/         <- annotated output images appear here
└── detections.json          <- detection results appear here
```

## Parameters (in `config/config.yaml`)

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

## Quick checks

- **0 detections** → check `raw_model_output` in `detections.json`; if boxes
  are present but `weed_count` is 0, lower `min_box_area_fraction`.
- **Too many boxes** → lower `containment_threshold` and/or raise
  `min_box_area_fraction`.
- **Slow / still looping** → raise `repetition_penalty` or lower
  `no_repeat_ngram_size`.