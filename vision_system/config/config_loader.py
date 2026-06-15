"""
Config loader for the weed detection pipeline.

Loads config/config.yaml and resolves string-based settings (dtype, font,
paths) into the actual Python objects the main script needs.
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
    """
    Load config.yaml and resolve it into a flat dict of ready-to-use values.

    Paths are resolved relative to the project root (parent of config/).
    """
    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    project_root = config_path.resolve().parent.parent

    cfg = {}

    # --- model ---
    cfg["MODEL_ID"] = raw["model"]["model_id"]
    cfg["DEVICE"] = raw["model"]["device"]
    dtype_str = raw["model"]["dtype"]
    if dtype_str not in _DTYPE_MAP:
        raise ValueError(
            f"Unknown dtype '{dtype_str}' in config.yaml. "
            f"Valid options: {list(_DTYPE_MAP.keys())}"
        )
    cfg["DTYPE"] = _DTYPE_MAP[dtype_str]

    # --- paths ---
    cfg["INPUT_DIR"] = project_root / raw["paths"]["input_dir"]
    cfg["OUTPUT_DIR"] = project_root / raw["paths"]["output_dir"]
    cfg["RESULTS_FILE"] = project_root / raw["paths"]["results_file"]

    # --- detection ---
    cfg["DETECTION_PROMPTS"] = raw["detection"]["prompts"]
    cfg["SUPPORTED_EXTENSIONS"] = set(raw["detection"]["supported_extensions"])
    cfg["CONF_THRESHOLD"] = raw["detection"]["conf_threshold"]

    # --- generation ---
    cfg["MAX_NEW_TOKENS"] = raw["generation"]["max_new_tokens"]
    cfg["REPETITION_PENALTY"] = raw["generation"]["repetition_penalty"]
    cfg["NO_REPEAT_NGRAM_SIZE"] = raw["generation"]["no_repeat_ngram_size"]

    # --- filtering ---
    cfg["MIN_BOX_AREA_FRACTION"] = raw["filtering"]["min_box_area_fraction"]
    cfg["CONTAINMENT_THRESHOLD"] = raw["filtering"]["containment_threshold"]

    # --- visualization ---
    cfg["BOX_COLOR"] = tuple(raw["visualization"]["box_color_bgr"])
    cfg["BOX_THICKNESS"] = raw["visualization"]["box_thickness"]
    cfg["LABEL_COLOR"] = tuple(raw["visualization"]["label_color_bgr"])
    font_str = raw["visualization"]["font"]
    if font_str not in _FONT_MAP:
        raise ValueError(
            f"Unknown font '{font_str}' in config.yaml. "
            f"Valid options: {list(_FONT_MAP.keys())}"
        )
    cfg["FONT"] = _FONT_MAP[font_str]
    cfg["FONT_SCALE"] = raw["visualization"]["font_scale"]

    return cfg