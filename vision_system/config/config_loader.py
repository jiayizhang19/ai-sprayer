"""
Config loader - Clean & user-friendly model switching
"""

from pathlib import Path

import torch
import cv2
import yaml

CONFIG_PATH = Path(__file__).resolve().parent / "config.yaml"

_DTYPE_MAP = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}

_FONT_MAP = {
    "FONT_HERSHEY_SIMPLEX": cv2.FONT_HERSHEY_SIMPLEX,
    "FONT_HERSHEY_PLAIN": cv2.FONT_HERSHEY_PLAIN,
    "FONT_HERSHEY_DUPLEX": cv2.FONT_HERSHEY_DUPLEX,
    "FONT_HERSHEY_COMPLEX": cv2.FONT_HERSHEY_COMPLEX,
    "FONT_HERSHEY_TRIPLEX": cv2.FONT_HERSHEY_TRIPLEX,
}


def load_config(config_path: Path = CONFIG_PATH) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    project_root = config_path.resolve().parent.parent
    cfg = {}

    # --- Model Configuration ---
    model_section = raw["model"]
    cfg["MODEL_TYPE"] = model_section["model_type"].lower()

    cfg["LOCATEANYTHING_ID"] = model_section.get("locateanything_id", "nvidia/LocateAnything-3B")
    cfg["YOLO_MODEL"] = model_section.get("yolo_model", "yolov8s.pt")

    cfg["DEVICE"] = model_section["device"]
    dtype_str = model_section["dtype"]
    cfg["DTYPE"] = _DTYPE_MAP[dtype_str]

    # --- Evaluation Block ---
    eval_section = raw.get("evaluation", {})
    cfg["VISUALIZE_DETECTIONS"] = eval_section.get("visualize_detections", True)

    # --- Paths ---
    cfg["INPUT_DIR"] = project_root / raw["paths"]["input_dir"]
    cfg["OUTPUT_DIR"] = project_root / raw["paths"]["output_dir"]
    cfg["RESULTS_FILE"] = project_root / raw["paths"]["results_file"]

    # --- Detection & Generation (shared) ---
    cfg["DETECTION_PROMPTS"] = raw["detection"]["prompts"]
    cfg["SUPPORTED_EXTENSIONS"] = set(raw["detection"]["supported_extensions"])
    cfg["CONF_THRESHOLD"] = raw["detection"]["conf_threshold"]

    cfg["MAX_NEW_TOKENS"] = raw["generation"]["max_new_tokens"]
    cfg["REPETITION_PENALTY"] = raw["generation"]["repetition_penalty"]
    cfg["NO_REPEAT_NGRAM_SIZE"] = raw["generation"]["no_repeat_ngram_size"]

    # --- Filtering ---
    cfg["MIN_BOX_AREA_FRACTION"] = raw["filtering"]["min_box_area_fraction"]
    cfg["CONTAINMENT_THRESHOLD"] = raw["filtering"]["containment_threshold"]

    # --- Visualization ---
    cfg["BOX_COLOR"] = tuple(raw["visualization"]["box_color_bgr"])
    cfg["BOX_THICKNESS"] = raw["visualization"]["box_thickness"]
    cfg["FONT"] = _FONT_MAP[raw["visualization"]["font"]]
    cfg["FONT_SCALE"] = raw["visualization"]["font_scale"]

    return cfg