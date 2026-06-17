"""
Step 1: Zero-shot weed detection using NVIDIA LocateAnything-3B
---------------------------------------------------------------
Final robust version with:
- Fixed attention implementation
- Fixed output decoding
- Centroid visualization
- Improved label cleaning
"""

import sys
import json
import re
import time
from pathlib import Path

import torch
import cv2
import numpy as np
from PIL import Image
from transformers import AutoProcessor, AutoModel

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.config_loader import load_config

# ─── CONFIG ──────────────────────────────────────────────────────────────────
cfg = load_config()

MODEL_ID   = cfg["MODEL_ID"]
DEVICE     = cfg["DEVICE"]
DTYPE      = cfg["DTYPE"]

INPUT_DIR    = cfg["INPUT_DIR"]
OUTPUT_DIR   = cfg["OUTPUT_DIR"]
RESULTS_FILE = cfg["RESULTS_FILE"]

DETECTION_PROMPTS    = cfg["DETECTION_PROMPTS"]
SUPPORTED_EXTENSIONS = cfg["SUPPORTED_EXTENSIONS"]

MAX_NEW_TOKENS        = cfg["MAX_NEW_TOKENS"]
REPETITION_PENALTY    = cfg["REPETITION_PENALTY"]
NO_REPEAT_NGRAM_SIZE  = cfg["NO_REPEAT_NGRAM_SIZE"]

MIN_BOX_AREA_FRACTION = cfg["MIN_BOX_AREA_FRACTION"]
CONTAINMENT_THRESHOLD = cfg["CONTAINMENT_THRESHOLD"]

BOX_COLOR     = cfg["BOX_COLOR"]
BOX_THICKNESS = cfg["BOX_THICKNESS"]
FONT          = cfg["FONT"]
FONT_SCALE    = cfg["FONT_SCALE"]


# ─── MODEL LOADING ────────────────────────────────────────────────────────────
def load_model():
    print(f"Loading model: {MODEL_ID}")
    print(f"Device: {DEVICE} | Dtype: {DTYPE}")
    print("This may take a few minutes...\n")

    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)

    model = AutoModel.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        torch_dtype=DTYPE,
        attn_implementation="sdpa",
    )

    # Force correct attention implementation
    for obj in [model, getattr(model, "language_model", None),
                getattr(getattr(model, "language_model", None), "model", None)]:
        if obj is None:
            continue
        if hasattr(obj, "_attn_implementation"):
            obj._attn_implementation = "sdpa"
        if hasattr(obj, "config"):
            if hasattr(obj.config, "_attn_implementation"):
                obj.config._attn_implementation = "sdpa"
            if hasattr(obj.config, "attn_implementation"):
                obj.config.attn_implementation = "sdpa"

    if DEVICE == "cuda":
        model = model.to(DEVICE)
    model.eval()

    print("Model loaded successfully.\n")
    return model, processor


# ─── LABEL CLEANING ───────────────────────────────────────────────────────────
def clean_label(label: str) -> str:
    label = label.strip().lower()
    if not label:
        return "weed"
    for word in ["brome", "weed", "chickweed"]:
        if word in label:
            return word
    n = len(label)
    for unit_len in range(1, n//2 + 2):
        unit = label[:unit_len]
        if unit * (n // unit_len) in label:
            if len(unit) >= 3:
                return unit
    return label


# ─── NMS ─────────────────────────────────────────────────────────────────────
def apply_nms(detections: list[dict], iou_threshold: float = 0.45) -> list[dict]:
    if len(detections) <= 1:
        return detections
    boxes = np.array([[d["x1"], d["y1"], d["x2"], d["y2"]] for d in detections])
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    order = areas.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(boxes[i, 0], boxes[order[1:], 0])
        yy1 = np.maximum(boxes[i, 1], boxes[order[1:], 1])
        xx2 = np.minimum(boxes[i, 2], boxes[order[1:], 2])
        yy2 = np.minimum(boxes[i, 3], boxes[order[1:], 3])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        inds = np.where(ovr <= iou_threshold)[0]
        order = order[inds + 1]
    return [detections[i] for i in keep]


# ─── FILTERING & PARSING ─────────────────────────────────────────────────────
def filter_detections(detections: list[dict], img_w: int, img_h: int) -> list[dict]:
    img_area = img_w * img_h
    min_area = MIN_BOX_AREA_FRACTION * img_area
    sized = [(max(0, d["x2"]-d["x1"])*max(0, d["y2"]-d["y1"]), d) for d in detections 
             if max(0, d["x2"]-d["x1"])*max(0, d["y2"]-d["y1"]) >= min_area]
    sized.sort(key=lambda x: x[0], reverse=True)
    accepted = []
    for _, det in sized:
        if not any(
            (min(det["x2"], k["x2"]) - max(det["x1"], k["x1"])) * 
            (min(det["y2"], k["y2"]) - max(det["y1"], k["y1"])) / 
            max(1e-6, (det["x2"]-det["x1"])*(det["y2"]-det["y1"])) >= CONTAINMENT_THRESHOLD
            for k in accepted
        ):
            accepted.append(det)
    return accepted


def parse_boxes(response_text: str, img_w: int, img_h: int) -> list[dict]:
    detections = []
    ref_blocks = re.findall(r'<ref>(.*?)</ref>((?:<box>(?:<\d+>){4}</box>)+)', response_text, re.DOTALL)
    for label, raw in ref_blocks:
        label = clean_label(label)
        raw_boxes = re.findall(r'<box><(\d+)><(\d+)><(\d+)><(\d+)></box>', raw)
        last_box = None
        for bx in raw_boxes:
            x1n, y1n, x2n, y2n = map(int, bx)
            if last_box == (x1n, y1n, x2n, y2n):
                continue
            last_box = (x1n, y1n, x2n, y2n)
            x1 = int(x1n / 1000 * img_w)
            y1 = int(y1n / 1000 * img_h)
            x2 = int(x2n / 1000 * img_w)
            y2 = int(y2n / 1000 * img_h)
            detections.append({"label": label, "x1": x1, "y1": y1, "x2": x2, "y2": y2})
    filtered = filter_detections(detections, img_w, img_h)
    return apply_nms(filtered)


# ─── DETECTION ───────────────────────────────────────────────────────────────
def detect_weeds(image: Image.Image, model, processor, prompt: str):
    img_w, img_h = image.size

    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]

    text_input = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    inputs = processor(text=text_input, images=[image], return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            tokenizer=processor.tokenizer,
            use_cache=True,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            repetition_penalty=REPETITION_PENALTY,
            no_repeat_ngram_size=NO_REPEAT_NGRAM_SIZE,
        )

    # Robust decoding - handle both tensor and string output
    if isinstance(output, str):
        raw_text = output
    else:
        # Tensor case
        generated_tokens = output[0, inputs["input_ids"].shape[1]:]
        raw_text = processor.decode(generated_tokens, skip_special_tokens=True)

    detections = parse_boxes(raw_text, img_w, img_h)
    return detections, raw_text


# ─── VISUALISATION WITH CENTROID ─────────────────────────────────────────────
def draw_detections(image_bgr: np.ndarray, detections: list[dict]) -> np.ndarray:
    annotated = image_bgr.copy()
    for i, det in enumerate(detections):
        x1, y1, x2, y2 = det["x1"], det["y1"], det["x2"], det["y2"]
        label = f"{det['label']} #{i+1}"

        cv2.rectangle(annotated, (x1, y1), (x2, y2), BOX_COLOR, BOX_THICKNESS)

        (tw, th), _ = cv2.getTextSize(label, FONT, FONT_SCALE, 1)
        cv2.rectangle(annotated, (x1, y1 - th - 6), (x1 + tw + 4, y1), BOX_COLOR, -1)
        cv2.putText(annotated, label, (x1 + 2, y1 - 4), FONT, FONT_SCALE, (0, 0, 0), 1)

        # Centroid
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        cv2.circle(annotated, (cx, cy), 6, (0, 0, 255), -1)
        cv2.circle(annotated, (cx, cy), 8, (0, 255, 255), 2)

        coord_text = f"({cx},{cy})"
        (twc, _), _ = cv2.getTextSize(coord_text, FONT, 0.5, 1)
        text_y = cy + 25 if cy < annotated.shape[0] - 40 else cy - 15
        cv2.putText(annotated, coord_text, (cx - twc//2, text_y), FONT, 0.5, (0, 0, 255), 1)

    cv2.putText(annotated, f"Weeds detected: {len(detections)}", (10, 30),
                FONT, 0.9, (0, 255, 255), 2)
    return annotated


# ─── MAIN ────────────────────────────────────────────────────────────────────
def process_folder(input_dir: str, output_dir: str, model, processor):
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    image_files = sorted(f for f in input_path.iterdir() if f.suffix.lower() in SUPPORTED_EXTENSIONS)
    if not image_files:
        print("No images found!")
        return

    prompt = DETECTION_PROMPTS[0] if DETECTION_PROMPTS else "Locate all brome plants separately."
    print(f"Using prompt: \"{prompt}\"\n")

    all_results = []
    for idx, img_path in enumerate(image_files):
        print(f"[{idx+1}/{len(image_files)}] Processing: {img_path.name}")

        pil_image = Image.open(img_path).convert("RGB")
        bgr_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

        t_start = time.time()
        detections, raw_text = detect_weeds(pil_image, model, processor, prompt)
        elapsed = time.time() - t_start

        print(f"  → {len(detections)} weed(s) detected ({elapsed:.1f}s)")

        annotated = draw_detections(bgr_image, detections)
        out_path = output_path / f"detected_{img_path.stem}.jpg"
        cv2.imwrite(str(out_path), annotated)

        all_results.append({
            "image": img_path.name,
            "weed_count": len(detections),
            "detections": detections,
            "raw_model_output": raw_text,
            "inference_time_seconds": round(elapsed, 2),
        })

        print(f"  → Saved: {out_path.name}\n")

    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump({"results": all_results}, f, indent=2)

    print("✅ Done! All images processed.")


if __name__ == "__main__":
    model, processor = load_model()
    process_folder(INPUT_DIR, OUTPUT_DIR, model, processor)