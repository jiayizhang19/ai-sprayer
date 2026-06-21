"""
Modular Weed Detection Pipeline
------------------------------
Generates standardized classification, bounding box, and centroid coordinate data
for downstream ROS2 consumption. Reads strictly from the production 'model' block.
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

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.config_loader import load_config

cfg = load_config()

# ─── CONFIG & PATHS ──────────────────────────────────────────────────────────
RESULTS_FILE = cfg["RESULTS_FILE"]
RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)

MODEL_TYPE = cfg["MODEL_TYPE"]
LOCATEANYTHING_ID = cfg["LOCATEANYTHING_ID"]
YOLO_MODEL = cfg["YOLO_MODEL"]  # Reads strictly from primary production block

INPUT_DIR = cfg["INPUT_DIR"]
OUTPUT_DIR = cfg["OUTPUT_DIR"]
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DETECTION_PROMPTS = cfg["DETECTION_PROMPTS"]
SUPPORTED_EXTENSIONS = cfg["SUPPORTED_EXTENSIONS"]

MAX_NEW_TOKENS = cfg["MAX_NEW_TOKENS"]
REPETITION_PENALTY = cfg["REPETITION_PENALTY"]
NO_REPEAT_NGRAM_SIZE = cfg["NO_REPEAT_NGRAM_SIZE"]

MIN_BOX_AREA_FRACTION = cfg["MIN_BOX_AREA_FRACTION"]
CONTAINMENT_THRESHOLD = cfg["CONTAINMENT_THRESHOLD"]

# Extracted dynamically from config.yaml under the evaluation block
VISUALIZE_DETECTIONS = cfg["VISUALIZE_DETECTIONS"]


# ─── MODEL LOADING ───────────────────────────────────────────────────────────
def load_model(model_override_path: str = None):
    """
    Loads the designated vision model. Defaults strictly to the production config, 
    but accepts explicit structural path injections from evaluation layers.
    """
    if MODEL_TYPE == "yolo":
        target_path = model_override_path if model_override_path else YOLO_MODEL
        path_obj = Path(target_path)
        
        # Pull run folder parent layout name dynamically for trace clarity
        if path_obj.parent.name == "weights":
            yolo_display_name = path_obj.parent.parent.name
        else:
            yolo_display_name = path_obj.stem
            
        print(f"Loading Model Setup [YOLO: {yolo_display_name}]...")
        from ultralytics import YOLO
        return YOLO(target_path), None
    else:
        target_id = model_override_path if model_override_path else LOCATEANYTHING_ID
        print(f"Loading Model Setup [{MODEL_TYPE.upper()}: {target_id}]...")
        from transformers import AutoProcessor, AutoModelForCausalLM
        processor = AutoProcessor.from_pretrained(target_id, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            target_id,
            trust_remote_code=True,
            torch_dtype=cfg["DTYPE"]
        ).to(cfg["DEVICE"])
        return model, processor


# ─── INFERENCE: YOLO ─────────────────────────────────────────────────────────
def detect_with_yolo(pil_image: Image.Image, model) -> tuple[list[dict], str]:
    img_cv = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    results = model(img_cv, verbose=False)
    
    detections = []
    if not results:
        return detections, "No results returned"
        
    res = results[0]
    names = res.names
    
    for box in res.boxes:
        coords = box.xyxy[0].tolist()
        x1, y1, x2, y2 = map(int, coords)
        conf = float(box.conf[0])
        cls_id = int(box.cls[0])
        class_name = names.get(cls_id, f"UNKNOWN_{cls_id}")
        
        detections.append({
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            "confidence": conf,
            "label": class_name
        })
        
    return detections, str(res)


# ─── INFERENCE: LOCATEANYTHING ───────────────────────────────────────────────
def detect_with_locateanything(pil_image: Image.Image, model, processor, prompt: str) -> tuple[list[dict], str]:
    w, h = pil_image.size
    
    inputs = processor(images=pil_image, text=prompt, return_tensors="pt")
    inputs = {k: v.to(cfg["DEVICE"]) for k, v in inputs.items()}
    if "pixel_values" in inputs:
        inputs["pixel_values"] = inputs["pixel_values"].to(cfg["DTYPE"])

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            repetition_penalty=REPETITION_PENALTY,
            no_repeat_ngram_size=NO_REPEAT_NGRAM_SIZE,
            do_sample=False
        )

    generated_text = processor.decode(generated_ids[0], skip_special_tokens=True)
    
    pattern = r"([^\[,\n]+)\s*\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]"
    matches = re.findall(pattern, generated_text)
    
    detections = []
    for match in matches:
        label = match[0].strip()
        y1 = int(float(match[1]) / 1000.0 * h)
        x1 = int(float(match[2]) / 1000.0 * w)
        y2 = int(float(match[3]) / 1000.0 * h)
        x2 = int(float(match[4]) / 1000.0 * w)
        
        detections.append({
            "x1": min(x1, x2), "y1": min(y1, y2),
            "x2": max(x1, x2), "y2": max(y1, y2),
            "confidence": 1.0,
            "label": label
        })
        
    return detections, generated_text


# ─── BOUNDING BOX GEOMETRY POST-PROCESSING ───────────────────────────────────
def post_process_detections(detections: list[dict], img_w: int, img_h: int) -> list[dict]:
    valid = []
    img_area = img_w * img_h
    
    for d in detections:
        box_area = (d["x2"] - d["x1"]) * (d["y2"] - d["y1"])
        if (box_area / img_area) >= MIN_BOX_AREA_FRACTION:
            valid.append(d)
            
    valid = sorted(valid, key=lambda x: (x["x2"] - x["x1"]) * (x["y2"] - x["y1"]), reverse=True)
    keep = []
    
    for i, box in enumerate(valid):
        is_contained = False
        b_area = (box["x2"] - box["x1"]) * (box["y2"] - box["y1"])
        
        for accepted in keep:
            ix1 = max(box["x1"], accepted["x1"])
            iy1 = max(box["y1"], accepted["y1"])
            ix2 = min(box["x2"], accepted["x2"])
            iy2 = min(box["y2"], accepted["y2"])
            
            if ix2 > ix1 and iy2 > iy1:
                inter_area = (ix2 - ix1) * (iy2 - iy1)
                if (inter_area / b_area) > CONTAINMENT_THRESHOLD:
                    is_contained = True
                    break
        if not is_contained:
            keep.append(box)
            
    return keep


# ─── VISUALIZATION GRAPHICS FUNCTION ─────────────────────────────────────────
def save_annotated_image(img_path: Path, ros_detections: list[dict]):
    """Draws green bounding boxes and matching green high-contrast tracking strings on verification frames."""
    img_cv = cv2.imread(str(img_path))
    if img_cv is None:
        return
        
    BOX_COLOR = (0, 255, 0)       # Bright Green boundary
    TEXT_COLOR = (0, 255, 0)      # Bright Green text labels
    BG_CUSHION = (255, 255, 255)  # White background fill backdrop block
    
    for d in ros_detections:
        bbox = d["bbox"]
        cx, cy = d["centroid_x"], d["centroid_y"]
        
        # 1. Draw boundary box
        cv2.rectangle(img_cv, (bbox["x1"], bbox["y1"]), (bbox["x2"], bbox["y2"]), BOX_COLOR, 2)
        
        # 2. Draw centroid tracker crosshair tags
        cv2.circle(img_cv, (cx, cy), 6, (0, 0, 255), -1)
        cv2.circle(img_cv, (cx, cy), 2, (255, 255, 255), -1)
        
        # 3. Handle high-contrast text overlays
        label_str = f"{d['class']} ({d['confidence']:.2f}) [Cx:{cx}, Cy:{cy}]"
        
        # Compute exact pixel space dimensions of the string block
        (tw, th), baseline = cv2.getTextSize(label_str, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        
        text_x = bbox["x1"]
        text_y = bbox["y1"] - 8 if bbox["y1"] - 8 > th + baseline else bbox["y1"] + th + 8
        
        # Render backdrop contrast anchor box
        cv2.rectangle(
            img_cv, 
            (text_x, text_y - th - 4), 
            (text_x + tw + 4, text_y + baseline), 
            BG_CUSHION, 
            -1
        )
        
        # Lay text string cleanly over the cushion
        cv2.putText(
            img_cv, 
            label_str, 
            (text_x + 2, text_y - 2),
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.6, 
            TEXT_COLOR, 
            1, 
            cv2.LINE_AA
        )
                    
    out_path = OUTPUT_DIR / f"detected_{img_path.name}"
    cv2.imwrite(str(out_path), img_cv)


# ─── MAIN ROS2 COMPATIBLE EXECUTOR ───────────────────────────────────────────
def main():
    # Production utilizes default configurations (e.g. ep30 weights)
    model, processor = load_model()
    
    image_files = sorted(f for f in INPUT_DIR.iterdir() if f.suffix.lower() in SUPPORTED_EXTENSIONS)
    prompt = DETECTION_PROMPTS[0] if DETECTION_PROMPTS else "Locate all individual weed plants."

    print(f"Target Images: {len(image_files)} found in {INPUT_DIR}")
    
    output_results = []

    for idx, img_path in enumerate(image_files):
        print(f"[{idx+1}/{len(image_files)}] Processing: {img_path.name}")
        
        pil_image = Image.open(img_path).convert("RGB")
        img_w, img_h = pil_image.size
        
        t0 = time.time()
        if MODEL_TYPE == "yolo":
            detections, raw_output = detect_with_yolo(pil_image, model)
        else:
            detections, raw_output = detect_with_locateanything(pil_image, model, processor, prompt)
        elapsed = time.time() - t0

        filtered_detections = post_process_detections(detections, img_w, img_h)

        # Structure normalized coordinate targets explicitly for downstream ROS nodes
        ros_detections = []
        for d in filtered_detections:
            cx = int(round((d["x1"] + d["x2"]) / 2.0))
            cy = int(round((d["y1"] + d["y2"]) / 2.0))
            
            ros_detections.append({
                "class": d["label"].strip().upper(),
                "centroid_x": cx,
                "centroid_y": cy,
                "bbox": {
                    "x1": d["x1"],
                    "y1": d["y1"],
                    "x2": d["x2"],
                    "y2": d["y2"]
                },
                "confidence": d["confidence"]
            })

        # Check configuration value
        if VISUALIZE_DETECTIONS:
            save_annotated_image(img_path, ros_detections)

        output_results.append({
            "image": img_path.name,
            "weed_count": len(ros_detections),
            "detections": ros_detections,
            "raw_output": raw_output[:500],
            "inference_time_seconds": round(elapsed, 2)
        })

    # Save detections payload to configured results path (project root by default)
    final_output_file = RESULTS_FILE
    
    final_payload = {
        "config": {
            "model_type": MODEL_TYPE,
            "save_annotated_images": VISUALIZE_DETECTIONS,
            "device": cfg["DEVICE"],
            "dtype": str(cfg["DTYPE"]).replace("torch.", "")
        },
        "results": output_results
    }

    with open(final_output_file, "w", encoding="utf-8") as f:
        json.dump(final_payload, f, indent=2)

    print(f"\n✅ Detection completed safely. Structured payload exported to: {final_output_file}")


if __name__ == "__main__":
    main()