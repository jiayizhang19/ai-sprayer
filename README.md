# Weed Detection — Parameter Tuning Guide

`weed_detection_step1.py` runs zero-shot weed detection using LocateAnything-3B.
All parameters below are in the `# ─── CONFIG ───` section (or in the
`model.generate()` call inside `detect_weeds`).

| Parameter | Default | Adjust when... |
|---|---|---|
| `DEVICE` | `"cpu"` | You have an NVIDIA GPU available → set to `"cuda"` |
| `DTYPE` | `torch.float32` | On GPU → set to `torch.float16` for faster inference |
| `INPUT_DIR` | `./weed_images` | Point at your image folder |
| `OUTPUT_DIR` | `./weed_detections` | Change where annotated images are saved |
| `RESULTS_FILE` | `./detections.json` | Change where detection results are saved |
| `DETECTION_PROMPTS` | `"Locate all weeds in this image."` | Results are noisy or missing certain weed types → try a more specific prompt |
| `MIN_BOX_AREA_FRACTION` | `0.005` | Small junk boxes still appear → **increase**. Real small/young weeds get dropped → **decrease** |
| `CONTAINMENT_THRESHOLD` | `0.9` | Junk boxes overlapping real detections aren't removed → **decrease** (e.g. 0.8). Adjacent real weeds get merged into one box → **increase** (closer to 1.0) |
| `repetition_penalty` | `1.3` | Model still loops/repeats boxes → **increase** (e.g. 1.5). Model misses legitimately repeated detections → **decrease** |
| `no_repeat_ngram_size` | `8` | Loops persist → **decrease** (e.g. 4–6). Output cut off too early → **increase** |
| `max_new_tokens` | `512` | Many weeds per image and output gets cut off → **increase** |

## Quick checks

- **0 detections** → check `raw_model_output` in `detections.json`; if boxes
  are present but `weed_count` is 0, lower `MIN_BOX_AREA_FRACTION`.
- **Too many boxes** → lower `CONTAINMENT_THRESHOLD` and/or raise
  `MIN_BOX_AREA_FRACTION`.
- **Slow / still looping** → raise `repetition_penalty` or lower
  `no_repeat_ngram_size`.
