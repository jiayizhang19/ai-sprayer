"""
Modular Weed Detection Pipeline
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.config_loader import load_config

cfg = load_config()

# ─── CONFIG ──────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_ROOT = PROJECT_ROOT / "yolo_vs_locateanything"
RESULTS_ROOT.mkdir(parents=True, exist_ok=True)

MODEL_TYPE = cfg["MODEL_TYPE"]
LOCATEANYTHING_ID = cfg["LOCATEANYTHING_ID"]
YOLO_MODEL = cfg["YOLO_MODEL"]

INPUT_DIR = cfg["INPUT_DIR"]
OUTPUT_DIR = cfg["OUTPUT_DIR"]

DETECTION_PROMPTS = cfg["DETECTION_PROMPTS"]
SUPPORTED_EXTENSIONS = cfg["SUPPORTED_EXTENSIONS"]

SAVE_ANNOTATED_IMAGES = cfg.get("SAVE_ANNOTATED_IMAGES", True)

MAX_NEW_TOKENS = cfg["MAX_NEW_TOKENS"]
REPETITION_PENALTY = cfg["REPETITION_PENALTY"]
NO_REPEAT_NGRAM_SIZE = cfg["NO_REPEAT_NGRAM_SIZE"]

MIN_BOX_AREA_FRACTION = cfg["MIN_BOX_AREA_FRACTION"]
CONTAINMENT_THRESHOLD = cfg["CONTAINMENT_THRESHOLD"]

BOX_COLOR = cfg["BOX_COLOR"]
BOX_THICKNESS = cfg["BOX_THICKNESS"]
FONT = cfg["FONT"]
FONT_SCALE = cfg["FONT_SCALE"]


# ─── CLASS DISPLAY ──────────────────────────────────────────────────────────
CLASS_MAP = {
    0: "ALOMY", 1: "ANGAR", 2: "APESV", 3: "ARTVU", 4: "AVEFA",
    5: "Brome (BROST)", 6: "BRSNN", 7: "CAPBP", 8: "CENCY", 9: "CHEAL",
    10: "CHYSE", 11: "CIRAR", 12: "CONAR", 13: "EPHHE", 14: "EPHPE",
    15: "EROCI", 16: "FUMOF", 17: "GALAP", 18: "GERMO", 19: "LAPCO",
    20: "LOLMU", 21: "LYCAR", 22: "MATCH", 23: "MATIN", 24: "MELNO",
    25: "MYOAR", 26: "PAPRH", 27: "PLALA", 28: "PLAMA", 29: "POAAN",
    30: "POLAV", 31: "POLCO", 32: "POLLA", 33: "POLPE", 34: "RUMCR",
    35: "SENVU", 36: "SINAR", 37: "SOLNI", 38: "SONOL", 39: "STEME",
    40: "THLAR", 41: "Urtur", 42: "VERAR", 43: "VERPE", 44: "VICHI",
    45: "VIOAR"
}

def get_display_label(raw_label):
    if raw_label is None:
        return "Unknown"
    label_str = str(raw_label).strip().lower()
    try:
        cid = int(label_str)
        return CLASS_MAP.get(cid, f"Class {cid}")
    except:
        pass
    if label_str in ["brost", "brome", "5"]:
        return "Brome (BROST)"
    if label_str in ["urtur", "41"]:
        return "Urtur"
    return label_str.capitalize()


# ─── MODEL LOADING ───────────────────────────────────────────────────────────
def load_model():
    print(f"Loading {MODEL_TYPE.upper()} model...")

    if MODEL_TYPE == "yolo":
        from ultralytics import YOLO
        model = YOLO(YOLO_MODEL)
        print(f"✅ YOLOv8 loaded: {YOLO_MODEL}")
        return model, None
    else:
        from transformers import AutoProcessor, AutoModel
        print("Loading LocateAnything-3B (this may take a while on first run)...")
        
        processor = AutoProcessor.from_pretrained(LOCATEANYTHING_ID, trust_remote_code=True)
        
        model = AutoModel.from_pretrained(
            LOCATEANYTHING_ID,
            trust_remote_code=True,
            torch_dtype=cfg["DTYPE"],
        )

        # Aggressive SDPA fix
        def force_sdpa(obj):
            if obj is None:
                return
            if hasattr(obj, '_attn_implementation'):
                obj._attn_implementation = "sdpa"
            if hasattr(obj, 'config') and hasattr(obj.config, '_attn_implementation'):
                obj.config._attn_implementation = "sdpa"
            if hasattr(obj, 'modules'):
                for submodule in obj.modules():
                    if hasattr(submodule, '_attn_implementation'):
                        submodule._attn_implementation = "sdpa"
                    if hasattr(submodule, 'config') and hasattr(submodule.config, '_attn_implementation'):
                        submodule.config._attn_implementation = "sdpa"

        force_sdpa(model)
        force_sdpa(getattr(model, 'language_model', None))

        if cfg["DEVICE"] == "cuda":
            model = model.to("cuda")
        
        model.eval()
        print("✅ LocateAnything-3B loaded successfully.")
        return model, processor


# ─── SHARED HELPERS ─────────────────────────────────────────────────────────
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
            return unit if len(unit) >= 3 else label
    return label


def draw_detections(image_bgr: np.ndarray, detections: list[dict]) -> np.ndarray:
    annotated = image_bgr.copy()
    for i, det in enumerate(detections):
        x1, y1, x2, y2 = det["x1"], det["y1"], det["x2"], det["y2"]
        raw_label = det.get('label', 'weed')
        display_label = get_display_label(raw_label)
        label = f"{display_label} #{i+1}"

        cv2.rectangle(annotated, (x1, y1), (x2, y2), BOX_COLOR, BOX_THICKNESS)

        (tw, th), _ = cv2.getTextSize(label, FONT, FONT_SCALE, 1)
        cv2.rectangle(annotated, (x1, y1 - th - 6), (x1 + tw + 4, y1), BOX_COLOR, -1)
        cv2.putText(annotated, label, (x1 + 2, y1 - 4), FONT, FONT_SCALE, (0, 0, 0), 1)

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


# ─── YOLO DETECTION ─────────────────────────────────────────────────────────
def detect_with_yolo(image: Image.Image, model):
    results = model(image, conf=0.25, verbose=False)[0]
    detections = []
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        conf = float(box.conf[0])
        cls = int(box.cls[0])
        label = results.names[cls]
        detections.append({"label": label, "x1": x1, "y1": y1, "x2": x2, "y2": y2, "confidence": conf})
    return detections, str(results)


# ─── LOCATEANYTHING HELPERS ─────────────────────────────────────────────────
def filter_detections(detections, img_w, img_h):
    img_area = img_w * img_h
    min_area = MIN_BOX_AREA_FRACTION * img_area
    sized = [(max(0, d["x2"]-d["x1"])*max(0, d["y2"]-d["y1"]), d) for d in detections 
             if max(0, d["x2"]-d["x1"])*max(0, d["y2"]-d["y1"]) >= min_area]
    sized.sort(key=lambda x: x[0], reverse=True)
    accepted = []
    for _, det in sized:
        if not any((min(det["x2"], k["x2"]) - max(det["x1"], k["x1"])) * 
                   (min(det["y2"], k["y2"]) - max(det["y1"], k["y1"])) / 
                   max(1e-6, (det["x2"]-det["x1"])*(det["y2"]-det["y1"])) >= CONTAINMENT_THRESHOLD 
                   for k in accepted):
            accepted.append(det)
    return accepted


def apply_nms(detections, iou_threshold=0.45):
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


def parse_boxes(response_text: str, img_w: int, img_h: int):
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


def detect_with_locateanything(image: Image.Image, model, processor, prompt: str):
    img_w, img_h = image.size
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]

    text_input = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    inputs = processor(text=text_input, images=[image], return_tensors="pt").to(cfg["DEVICE"])

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

    if isinstance(output, str):
        raw_text = output
    else:
        raw_text = processor.decode(output[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    detections = parse_boxes(raw_text, img_w, img_h)
    return detections, raw_text


# ─── MAIN ────────────────────────────────────────────────────────────────────
def process_folder(input_dir: str, output_dir: str, model, processor=None):
    # (This function is not used by detection_evaluation.py, so it's kept minimal)
    print("process_folder called - not used in evaluation mode.")
    pass


if __name__ == "__main__":
    model, processor = load_model()
    process_folder(INPUT_DIR, OUTPUT_DIR, model, processor)