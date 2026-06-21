"""
Evaluation Script: Model vs Ground Truth (Independent Execution)
-------------------------------------------------
Runs independent live inference on images using a configured evaluation model,
and scores the results against ground truth annotations.
"""

import sys
import json
from pathlib import Path
import yaml
import numpy as np
from PIL import Image

# ─── CONFIG & PATHS ──────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.config_loader import load_config
import weed_detection  # Reused for loading & inference logic

cfg = load_config()
IMAGES_DIR = cfg["INPUT_DIR"]

# Safely extract independent evaluation block from config.yaml
with open(PROJECT_ROOT / "config" / "config.yaml", "r", encoding="utf-8") as f:
    raw_yaml = yaml.safe_load(f)

eval_section = raw_yaml.get("evaluation", {})
EVAL_MODEL_TYPE = eval_section.get("eval_model_type", "yolo").lower()
EVAL_YOLO_MODEL = eval_section.get("eval_yolo_model", cfg["YOLO_MODEL"])
EVAL_LOCATEANYTHING_ID = eval_section.get("eval_locateanything_id", cfg["LOCATEANYTHING_ID"])

# Hot-patch weed_detection config variables dynamically to respect the eval settings
weed_detection.MODEL_TYPE = EVAL_MODEL_TYPE
weed_detection.YOLO_MODEL = EVAL_YOLO_MODEL
weed_detection.LOCATEANYTHING_ID = EVAL_LOCATEANYTHING_ID

# Ground truth setup
_input_dir_name = Path(cfg["INPUT_DIR"]).name  # e.g. "test" from "images/test"
_split_labels_dir = PROJECT_ROOT / "labels" / _input_dir_name
GT_LABELS_DIR = _split_labels_dir if _split_labels_dir.exists() else PROJECT_ROOT / "labels"

# Updated results root directory path name
RESULTS_ROOT = PROJECT_ROOT / "model_evaluation"
RESULTS_ROOT.mkdir(parents=True, exist_ok=True)

IOU_THRESHOLD = 0.5


# ─── EXPERIMENT NAME PARSING ────────────────────────────────────────────────
def get_experiment_model_name() -> str:
    """
    Extracts a short identifying name for the model file naming and configuration tracking.
    For YOLO weights path: extraction grabs the parent directory name of 'weights' (the run name).
    For LocateAnything HuggingFace ID: grabs the repository name string.
    """
    if EVAL_MODEL_TYPE == "yolo":
        path_obj = Path(EVAL_YOLO_MODEL)
        # If path looks like .../train/yolov8n_ep30_b4_lr0.0002/weights/best.pt
        if path_obj.parent.name == "weights":
            return path_obj.parent.parent.name
        return path_obj.stem
    else:
        # e.g., 'nvidia/LocateAnything-3B' -> 'LocateAnything-3B'
        return EVAL_LOCATEANYTHING_ID.split("/")[-1]

MODEL_EXP_NAME = get_experiment_model_name()

print(f"========================================================================")
print(f"Starting Independent Evaluation Workflow")
print(f"========================================================================")
print(f"Target Architecture: {EVAL_MODEL_TYPE.upper()}")
print(f"Target Experiment:   {MODEL_EXP_NAME}")
print(f"Target Model/Path:   {EVAL_YOLO_MODEL if EVAL_MODEL_TYPE == 'yolo' else EVAL_LOCATEANYTHING_ID}")
print(f"Images Path:        {IMAGES_DIR}")
print(f"Ground Truth Path:  {GT_LABELS_DIR}\n")


# ─── CLASS NAME RESOLUTION ───────────────────────────────────────────────────
def load_class_id_map() -> dict:
    data_yaml_path = PROJECT_ROOT / "config" / "data.yaml"
    if not data_yaml_path.exists():
        return {}
    with open(data_yaml_path, "r", encoding="utf-8") as f:
        data_cfg = yaml.safe_load(f)
    return {int(k): v for k, v in data_cfg.get("names", {}).items()}

CLASS_ID_MAP = load_class_id_map()


def label_matches_class(predicted_label: str, gt_class_code: str) -> bool:
    predicted_label = (predicted_label or "").strip().lower()
    gt_class_code = (gt_class_code or "").strip().lower()
    if not predicted_label or not gt_class_code:
        return False
    return predicted_label == gt_class_code or gt_class_code in predicted_label or predicted_label in gt_class_code


def load_ground_truth(image_name: str) -> list[dict]:
    stem = Path(image_name).stem.replace("detected_", "").replace("gt_", "")
    label_path = GT_LABELS_DIR / f"{stem}.txt"
    
    if not label_path.exists():
        for p in GT_LABELS_DIR.glob("*.txt"):
            if stem in p.stem or p.stem in stem:
                label_path = p
                break
        else:
            return []

    boxes = []
    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 5:
                continue
            class_id = int(parts[0])
            x_c, y_c, w, h = map(float, parts[1:])
            boxes.append({
                "class_id": class_id,
                "class_code": CLASS_ID_MAP.get(class_id, f"UNKNOWN_{class_id}"),
                "x_center": x_c, "y_center": y_c,
                "width": w, "height": h
            })
    return boxes


def yolo_to_pixel(box: dict, img_w: int, img_h: int) -> tuple:
    cx = box["x_center"] * img_w
    cy = box["y_center"] * img_h
    w = box["width"] * img_w
    h = box["height"] * img_h
    x1 = int(round(cx - w / 2))
    y1 = int(round(cy - h / 2))
    x2 = int(round(cx + w / 2))
    y2 = int(round(cy + h / 2))
    return max(0, x1), max(0, y1), min(img_w, x2), min(img_h, y2)


def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    if x2 < x1 or y2 < y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    if area1 == 0 or area2 == 0:
        return 0.0
    return inter / (area1 + area2 - inter)


def match_detections(gt_boxes: list[dict], pred_boxes: list[dict], iou_thresh: float = 0.5) -> dict:
    gt_matched = [False] * len(gt_boxes)
    matches = []
    per_class = {}

    def _bump(class_code: str, key: str):
        per_class.setdefault(class_code, {"tp": 0, "fp": 0, "fn": 0})
        per_class[class_code][key] += 1

    for p in pred_boxes:
        best_iou = 0.0
        best_idx = -1
        for i, gt in enumerate(gt_boxes):
            if gt_matched[i]:
                continue
            if not label_matches_class(p["label"], gt["class_code"]):
                continue
            iou = compute_iou(p["pixel_box"], gt["pixel_box"])
            if iou > best_iou:
                best_iou = iou
                best_idx = i

        if best_iou >= iou_thresh:
            gt_matched[best_idx] = True
            matched_class = gt_boxes[best_idx]["class_code"]
            matches.append((best_iou, True))
            _bump(matched_class, "tp")
        else:
            matches.append((0.0, False))
            _bump(p["label"].strip().upper() or "UNKNOWN", "fp")

    for i, gt in enumerate(gt_boxes):
        if not gt_matched[i]:
            _bump(gt["class_code"], "fn")

    tp = sum(1 for _, is_tp in matches if is_tp)
    fp = len(pred_boxes) - tp
    fn = len(gt_boxes) - tp
    ious = [iou for iou, _ in matches if iou > 0]

    return {
        "tp": tp, "fp": fp, "fn": fn,
        "matched_ious": ious,
        "precision": tp / (tp + fp) if (tp + fp) > 0 else 0.0,
        "recall": tp / (tp + fn) if (tp + fn) > 0 else 0.0,
        "per_class": per_class,
    }


# ─── LOCAL REPORTING UTILITY ─────────────────────────────────────────────────
def build_markdown_report(total_gt, total_pred, total_tp, total_fp, total_fn, 
                          precision, recall, f1, mean_iou, per_image_metrics, 
                          experiment_config, model_type, per_class_summary):
    """Generates local standalone markdown statistics without weed_detection dependencies."""
    lines = [
        f"# Evaluation Report: {model_type.upper()} ({MODEL_EXP_NAME})",
        "",
        "## Core Metrics Summary",
        f"- **Precision:** {precision:.4f}",
        f"- **Recall:** {recall:.4f}",
        f"- **F1 Score:** {f1:.4f}",
        f"- **Mean IoU (Matched):** {mean_iou:.4f}",
        "",
        "## Bounding Box Breakdown",
        f"- **Total Ground Truth Boxes:** {total_gt}",
        f"- **Total Predicted Boxes:** {total_pred}",
        f"- **True Positives (TP):** {total_tp}",
        f"- **False Positives (FP):** {total_fp}",
        f"- **False Negatives (FN):** {total_fn}",
        "",
        "## Per-Class Metrics Table",
        "| Class Code | TP | FP | FN | Precision | Recall | F1 Score |",
        "|--- |--- |--- |--- |--- |--- |--- |"
    ]
    for cls_code, metrics in per_class_summary.items():
        lines.append(
            f"| {cls_code} | {metrics['tp']} | {metrics['fp']} | {metrics['fn']} | "
            f"{metrics['precision']:.4f} | {metrics['recall']:.4f} | {metrics['f1']:.4f} |"
        )
    
    lines.extend([
        "",
        "## Setup Configuration Context",
        f"- **Model Identifier:** `{experiment_config.get('model_name')}`",
        f"- **Device Used:** `{experiment_config.get('device')}`",
        f"- **Data Type:** `{experiment_config.get('dtype')}`",
        ""
    ])
    return "\n".join(lines)


# ─── EVALUATION PIPELINE ─────────────────────────────────────────────────────
def main():
    # Load the requested model using weed_detection's backend structure
    model, processor = weed_detection.load_model()
    
    image_files = sorted(f for f in IMAGES_DIR.iterdir() if f.suffix.lower() in cfg["SUPPORTED_EXTENSIONS"])
    prompt = cfg["DETECTION_PROMPTS"][0] if cfg["DETECTION_PROMPTS"] else "Locate all brome plants separately."

    total_tp = total_fp = total_fn = 0
    all_ious = []
    per_image_metrics = []
    overall_per_class = {}

    for idx, img_path in enumerate(image_files):
        print(f"[{idx+1}/{len(image_files)}] Inferencing & Evaluating: {img_path.name}")
        
        pil_image = Image.open(img_path).convert("RGB")
        img_w, img_h = pil_image.size

        # Fetch predictions dynamically from the model instance
        if EVAL_MODEL_TYPE == "yolo":
            detections, _ = weed_detection.detect_with_yolo(pil_image, model)
        else:
            detections, _ = weed_detection.detect_with_locateanything(pil_image, model, processor, prompt)

        gt_norm = load_ground_truth(img_path.name)
        gt_boxes = [
            {"pixel_box": yolo_to_pixel(b, img_w, img_h), "class_code": b["class_code"]}
            for b in gt_norm
        ]
        
        # Unify outputs into evaluation bounding format
        pred_boxes = [
            {
                "pixel_box": (d["x1"], d["y1"], d["x2"], d["y2"]),
                "label": d.get("label", ""),
            }
            for d in detections
        ]

        metrics = match_detections(gt_boxes, pred_boxes, IOU_THRESHOLD)

        total_tp += metrics["tp"]
        total_fp += metrics["fp"]
        total_fn += metrics["fn"]
        all_ious.extend(metrics["matched_ious"])

        for class_code, counts in metrics["per_class"].items():
            overall_per_class.setdefault(class_code, {"tp": 0, "fp": 0, "fn": 0})
            for key in ("tp", "fp", "fn"):
                overall_per_class[class_code][key] += counts[key]

        f1 = 2 * metrics["precision"] * metrics["recall"] / (metrics["precision"] + metrics["recall"] + 1e-8)

        per_image_metrics.append({
            "image": img_path.name,
            "gt_count": len(gt_boxes),
            "pred_count": len(pred_boxes),
            "tp": metrics["tp"],
            "fp": metrics["fp"],
            "fn": metrics["fn"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": f1,
            "mean_iou_matched": np.mean(metrics["matched_ious"]) if metrics["matched_ious"] else 0.0
        })

    # Summary performance processing
    per_class_summary = {}
    for class_code, counts in overall_per_class.items():
        c_tp, c_fp, c_fn = counts["tp"], counts["fp"], counts["fn"]
        c_precision = c_tp / (c_tp + c_fp) if (c_tp + c_fp) > 0 else 0.0
        c_recall = c_tp / (c_tp + c_fn) if (c_tp + c_fn) > 0 else 0.0
        c_f1 = 2 * c_precision * c_recall / (c_precision + c_recall + 1e-8)
        per_class_summary[class_code] = {
            "tp": c_tp, "fp": c_fp, "fn": c_fn,
            "precision": c_precision, "recall": c_recall, "f1": c_f1,
        }

    total_gt = total_tp + total_fn
    total_pred = total_tp + total_fp
    precision = total_tp / total_pred if total_pred > 0 else 0.0
    recall = total_tp / total_gt if total_gt > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    mean_iou = np.mean(all_ious) if all_ious else 0.0

    # Include parsed experiment name directly into configuration data structure
    experiment_config = {
        "model_name": MODEL_EXP_NAME,
        "device": cfg["DEVICE"],
        "dtype": str(cfg["DTYPE"]).replace("torch.", ""),
        "prompt": prompt if EVAL_MODEL_TYPE == "locateanything" else None
    }
    
    # Generate files cleanly locally inside model_evaluation directory
    json_file = RESULTS_ROOT / f"eval_{MODEL_EXP_NAME}_results.json"
    md_file = RESULTS_ROOT / f"eval_{MODEL_EXP_NAME}_results.md"

    with open(json_file, "w", encoding="utf-8") as f:
        json.dump({
            "experiment_config": experiment_config,
            "overall": {
                "precision": precision, "recall": recall, "f1": f1, "mean_iou": mean_iou,
                "tp": total_tp, "fp": total_fp, "fn": total_fn, "total_gt": total_gt, "total_pred": total_pred
            },
            "per_class": per_class_summary,
            "per_image": per_image_metrics
        }, f, indent=2)

    md_report = build_markdown_report(
        total_gt, total_pred, total_tp, total_fp, total_fn,
        precision, recall, f1, mean_iou, per_image_metrics, experiment_config, EVAL_MODEL_TYPE,
        per_class_summary=per_class_summary
    )

    with open(md_file, "w", encoding="utf-8") as f:
        f.write(md_report)

    print(f"\n✅ Independent evaluation completed successfully!")
    print(f"   → Metrics JSON: {json_file.name}")
    print(f"   → Summary Report MD: {md_file.name}")


if __name__ == "__main__":
    main()