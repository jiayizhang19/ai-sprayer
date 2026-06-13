"""
Step 1: Zero-shot weed detection using NVIDIA LocateAnything-3B
---------------------------------------------------------------
Processes a folder of images and saves annotated results.
The detect_weeds() function is deliberately kept self-contained
so it can be dropped into a camera loop later with zero changes.

Hardware: CPU (default)
To switch to GPU later: change DEVICE = "cpu" -> DEVICE = "cuda"
"""

import os
import json
import re
import time
from pathlib import Path

import torch
import cv2
import numpy as np
from PIL import Image
from transformers import AutoProcessor, AutoModel

# ─── CONFIG ──────────────────────────────────────────────────────────────────

MODEL_ID   = "nvidia/LocateAnything-3B"
DEVICE     = "cpu"          # <-- change to "cuda" when you have a GPU
DTYPE      = torch.float32  # <-- change to torch.float16 on GPU for speed

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_DIR  = PROJECT_ROOT / "weed_images"      # folder with your weed images
OUTPUT_DIR = PROJECT_ROOT / "weed_detections"  # annotated images saved here
RESULTS_FILE = PROJECT_ROOT / "detections.json"  # all bounding boxes saved here

# What to ask the model. You can experiment with different prompts.
# More specific prompts can improve results for your crop type.
DETECTION_PROMPTS = [
    "Locate all weeds in this image.",
    # "Locate all broadleaf weeds between the crop rows.",  # try this if results are noisy
]

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Visual settings for drawn boxes
BOX_COLOR     = (0, 255, 0)   # green in BGR
BOX_THICKNESS = 2
LABEL_COLOR   = (0, 255, 0)
FONT          = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE    = 0.6
CONF_THRESHOLD = 0.0   # LocateAnything doesn't output confidence scores natively;
                        # set to 0 to accept all detections

# Post-processing filters for degenerate/junk boxes from repetition loops
MIN_BOX_AREA_FRACTION = 0.005   # box must cover at least 0.5% of image area to be kept
CONTAINMENT_THRESHOLD = 0.9     # if >=90% of a smaller box's area lies inside a larger
                                 # already-accepted box, the smaller one is dropped

# ─── MODEL LOADING ────────────────────────────────────────────────────────────

def load_model():
    """Load LocateAnything model and processor once at startup."""
    print(f"Loading model: {MODEL_ID}")
    print(f"Device: {DEVICE} | Dtype: {DTYPE}")
    print("This may take a few minutes on first run (downloading ~6GB)...\n")

    processor = AutoProcessor.from_pretrained(
        MODEL_ID,
        trust_remote_code=True
    )

    model = AutoModel.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        torch_dtype=DTYPE,
        attn_implementation="sdpa",
    )

    # Some LocateAnything remote-code versions ignore constructor attention args.
    for obj in [
        model,
        getattr(model, "language_model", None),
        getattr(getattr(model, "language_model", None), "model", None),
    ]:
        if obj is None:
            continue
        if hasattr(obj, "_attn_implementation"):
            obj._attn_implementation = "sdpa"
        if hasattr(obj, "config") and hasattr(obj.config, "_attn_implementation"):
            obj.config._attn_implementation = "sdpa"

    if DEVICE == "cuda":
        model = model.to(DEVICE)
    model.eval()

    print("Model loaded successfully.\n")
    return model, processor


# ─── DETECTION ───────────────────────────────────────────────────────────────

def filter_detections(detections: list[dict], img_w: int, img_h: int) -> list[dict]:
    """
    Remove junk boxes left over from generation degeneration loops.

    Two passes:
    1. Min-area filter: drop boxes smaller than MIN_BOX_AREA_FRACTION of the
       image area (the junk boxes tend to be tiny slivers).
    2. Containment filter: among the remaining boxes, sort largest-first and
       drop any box that is >=CONTAINMENT_THRESHOLD contained within a box
       already accepted (the junk boxes tend to nest inside the real ones).
    """
    img_area = img_w * img_h
    min_area = MIN_BOX_AREA_FRACTION * img_area

    # --- Pass 1: minimum area ---
    sized = []
    for det in detections:
        w = max(0, det["x2"] - det["x1"])
        h = max(0, det["y2"] - det["y1"])
        area = w * h
        if area >= min_area:
            sized.append((area, det))

    # --- Pass 2: containment, largest boxes first ---
    sized.sort(key=lambda t: t[0], reverse=True)

    accepted = []
    for area, det in sized:
        x1, y1, x2, y2 = det["x1"], det["y1"], det["x2"], det["y2"]

        is_contained = False
        for kept in accepted:
            kx1, ky1, kx2, ky2 = kept["x1"], kept["y1"], kept["x2"], kept["y2"]

            # Intersection
            ix1, iy1 = max(x1, kx1), max(y1, ky1)
            ix2, iy2 = min(x2, kx2), min(y2, ky2)
            iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
            inter_area = iw * ih

            if area > 0 and (inter_area / area) >= CONTAINMENT_THRESHOLD:
                is_contained = True
                break

        if not is_contained:
            accepted.append(det)

    return accepted


def parse_boxes(response_text: str, img_w: int, img_h: int) -> list[dict]:
    """
    Parse bounding boxes from LocateAnything text output.

    LocateAnything actually returns boxes in the format:
        <ref>label text</ref><box><x1><y1><x2><y2></box><box><x1><y1><x2><y2></box>...
    i.e. one <ref>...</ref> tag followed by one or more <box><a><b><c><d></box> tags,
    each number wrapped in its own angle brackets. Coordinates are normalised
    to a 0-1000 scale.

    The model sometimes degenerates into repeating the exact same box dozens
    of times (a known greedy-decoding repetition loop). Consecutive duplicate
    boxes are collapsed to a single detection here.

    Returns a list of dicts: {label, x1, y1, x2, y2, x1_norm, y1_norm, x2_norm, y2_norm}
    """
    detections = []

    # Match each <ref>label</ref> followed by a run of <box><a><b><c><d></box> tags
    ref_blocks = re.findall(r'<ref>(.*?)</ref>((?:<box>(?:<\d+>){4}</box>)+)', response_text, re.DOTALL)

    for label, raw in ref_blocks:
        label = label.strip()

        # Extract all 4 numbers per <box><x1><y1><x2><y2></box>
        raw_boxes = re.findall(r'<box><(\d+)><(\d+)><(\d+)><(\d+)></box>', raw)

        last_box = None
        for bx in raw_boxes:
            x1n, y1n, x2n, y2n = [int(v) for v in bx]

            # Skip consecutive exact-duplicate boxes (repetition-loop artifact)
            if last_box == (x1n, y1n, x2n, y2n):
                continue
            last_box = (x1n, y1n, x2n, y2n)

            # Denormalise from 0-1000 → pixel coordinates
            x1 = int(x1n / 1000 * img_w)
            y1 = int(y1n / 1000 * img_h)
            x2 = int(x2n / 1000 * img_w)
            y2 = int(y2n / 1000 * img_h)
            detections.append({
                "label": label,
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "x1_norm": x1n, "y1_norm": y1n, "x2_norm": x2n, "y2_norm": y2n,
            })

    return filter_detections(detections, img_w, img_h)


def detect_weeds(image: Image.Image, model, processor, prompt: str) -> tuple[list[dict], str]:
    """
    Core detection function — takes a PIL image, returns bounding boxes.

    This function is intentionally self-contained so it can be reused
    directly inside a real-time camera loop (Step 4) without changes.

    Args:
        image:     PIL.Image (RGB)
        model:     loaded LocateAnything model
        processor: loaded processor
        prompt:    natural language detection prompt

    Returns:
        detections: list of bounding box dicts
        raw_text:   raw model response (useful for debugging)
    """
    img_w, img_h = image.size

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt}
            ]
        }
    ]

    # Build model inputs
    text_input = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False
    )

    inputs = processor(
        text=text_input,
        images=[image],
        return_tensors="pt"
    ).to(DEVICE)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            tokenizer=processor.tokenizer,
            use_cache=True,
            max_new_tokens=512,
            do_sample=False,
            repetition_penalty=1.3,
            no_repeat_ngram_size=8,
        )

    # LocateAnything remote code returns decoded text directly.
    if isinstance(output, str):
        raw_text = output
    else:
        generated = output[:, inputs["input_ids"].shape[1]:]
        raw_text = processor.decode(generated[0], skip_special_tokens=True)

    detections = parse_boxes(raw_text, img_w, img_h)
    return detections, raw_text


# ─── VISUALISATION ───────────────────────────────────────────────────────────

def draw_detections(image_bgr: np.ndarray, detections: list[dict]) -> np.ndarray:
    """Draw bounding boxes and labels on a BGR image (OpenCV format)."""
    annotated = image_bgr.copy()

    for i, det in enumerate(detections):
        x1, y1, x2, y2 = det["x1"], det["y1"], det["x2"], det["y2"]
        label = f"{det['label']} #{i+1}"

        cv2.rectangle(annotated, (x1, y1), (x2, y2), BOX_COLOR, BOX_THICKNESS)

        # Label background for readability
        (tw, th), _ = cv2.getTextSize(label, FONT, FONT_SCALE, 1)
        cv2.rectangle(annotated, (x1, y1 - th - 6), (x1 + tw + 4, y1), BOX_COLOR, -1)
        cv2.putText(annotated, label, (x1 + 2, y1 - 4),
                    FONT, FONT_SCALE, (0, 0, 0), 1, cv2.LINE_AA)

    # Summary count
    count_text = f"Weeds detected: {len(detections)}"
    cv2.putText(annotated, count_text, (10, 30),
                FONT, 0.9, (0, 255, 255), 2, cv2.LINE_AA)

    return annotated


# ─── MAIN BATCH LOOP ─────────────────────────────────────────────────────────

def process_folder(input_dir: str, output_dir: str, model, processor):
    """
    Process all images in input_dir.
    Saves annotated images to output_dir and all results to detections.json.
    """
    input_path  = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    image_files = sorted([
        f for f in input_path.iterdir()
        if f.suffix.lower() in SUPPORTED_EXTENSIONS
    ])

    if not image_files:
        print(f"No images found in {input_dir}")
        print(f"Supported formats: {SUPPORTED_EXTENSIONS}")
        return

    print(f"Found {len(image_files)} image(s) in {input_dir}\n")
    prompt = DETECTION_PROMPTS[0]
    print(f"Detection prompt: \"{prompt}\"\n")
    print("-" * 60)

    all_results = []

    for idx, img_path in enumerate(image_files):
        print(f"[{idx+1}/{len(image_files)}] Processing: {img_path.name}")

        # Load image
        pil_image = Image.open(img_path).convert("RGB")
        bgr_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

        # Run detection
        t_start = time.time()
        detections, raw_text = detect_weeds(pil_image, model, processor, prompt)
        elapsed = time.time() - t_start

        print(f"  → {len(detections)} weed(s) detected  ({elapsed:.1f}s)")
        if detections:
            for i, d in enumerate(detections):
                print(f"     [{i+1}] {d['label']}  box=({d['x1']},{d['y1']}) → ({d['x2']},{d['y2']})")
        else:
            print(f"  → Raw model output: {raw_text[:200]}")  # debug if nothing detected

        # Draw and save annotated image
        annotated = draw_detections(bgr_image, detections)
        out_filename = output_path / f"detected_{img_path.stem}.jpg"
        cv2.imwrite(str(out_filename), annotated)

        # Accumulate results
        all_results.append({
            "image": img_path.name,
            "weed_count": len(detections),
            "detections": detections,
            "raw_model_output": raw_text,
            "inference_time_seconds": round(elapsed, 2),
        })

        print(f"  → Saved: {out_filename.name}")
        print()

    # Save JSON results
    with open(RESULTS_FILE, "w") as f:
        json.dump(all_results, f, indent=2)

    # Summary
    total_weeds = sum(r["weed_count"] for r in all_results)
    total_time  = sum(r["inference_time_seconds"] for r in all_results)
    print("=" * 60)
    print(f"Done! {len(image_files)} images processed.")
    print(f"Total weeds detected: {total_weeds}")
    print(f"Total inference time: {total_time:.1f}s  "
          f"(avg {total_time/len(image_files):.1f}s/image on {DEVICE})")
    print(f"Annotated images saved to: {output_dir}/")
    print(f"All results saved to:      {RESULTS_FILE}")


# ─── ENTRY POINT ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    model, processor = load_model()
    process_folder(INPUT_DIR, OUTPUT_DIR, model, processor)