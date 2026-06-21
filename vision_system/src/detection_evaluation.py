"""
Evaluation Script: Model vs Ground Truth
-------------------------------------------------
Checks for both yolo_detections.json and locateanything_detections.json
Generates corresponding evaluation reports for each.
"""

import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

# ─── CONFIG ──────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.config_loader import load_config

cfg = load_config()

# Use the SAME image folder that weed_detection.py read from when it
# generated the detections.json being evaluated here. Previously this was
# hardcoded to "weed_images", which silently broke evaluation whenever
# config.yaml's paths.input_dir pointed elsewhere (e.g. "images/test").
IMAGES_DIR = cfg["INPUT_DIR"]

# Ground truth labels are expected to live in a folder structure mirroring
# the images: if input_dir is "images/test", labels are expected in
# "labels/test" (Ultralytics-style layout), with a fallback to a flat
# "labels/" folder for setups that don't use split subfolders.
_input_dir_name = Path(cfg["INPUT_DIR"]).name  # e.g. "test" from "images/test"
_split_labels_dir = PROJECT_ROOT / "labels" / _input_dir_name
GT_LABELS_DIR = _split_labels_dir if _split_labels_dir.exists() else PROJECT_ROOT / "labels"

RESULTS_ROOT = PROJECT_ROOT / "yolo_vs_locateanything"

IOU_THRESHOLD = 0.5

print(f"Using IoU threshold = {IOU_THRESHOLD}")
print(f"Reading images from: {IMAGES_DIR}")
print(f"Reading ground truth labels from: {GT_LABELS_DIR}\n")


# ─── CLASS NAME RESOLUTION ───────────────────────────────────────────────────
def load_class_id_map() -> dict:
    """Maps ground truth class_id (int) -> class code (e.g. 5 -> 'BROST'),
    read from config/data.yaml so this stays in sync with however the
    dataset was labeled, instead of hardcoding the class list here."""
    import yaml
    data_yaml_path = PROJECT_ROOT / "config" / "data.yaml"
    if not data_yaml_path.exists():
        return {}
    with open(data_yaml_path, "r", encoding="utf-8") as f:
        data_cfg = yaml.safe_load(f)
    return {int(k): v for k, v in data_cfg.get("names", {}).items()}


CLASS_ID_MAP = load_class_id_map()


def label_matches_class(predicted_label: str, gt_class_code: str) -> bool:
    """
    True if a prediction's free-text/coded label refers to the same species
    as the ground truth class code. Case-insensitive substring match in
    both directions, since:
      - A correctly-trained YOLO model predicts the exact code (e.g. "BROST").
      - LocateAnything predicts free text (e.g. "brome"), which won't
        literally contain "BROST", so the code alone isn't enough — but for
        now we match on exact/substring code agreement; species-name
        synonyms (e.g. "brome") can be layered in via a synonym map if
        LocateAnything's output is also being evaluated here.
    """
    predicted_label = (predicted_label or "").strip().lower()
    gt_class_code = (gt_class_code or "").strip().lower()
    if not predicted_label or not gt_class_code:
        return False
    return predicted_label == gt_class_code or gt_class_code in predicted_label or predicted_label in gt_class_code


# ─── HELPERS (unchanged) ─────────────────────────────────────────────────────
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
    """
    Greedy IoU matching, now requiring BOTH IoU >= iou_thresh AND the
    predicted label matching the ground truth class for a true positive.

    gt_boxes:   list of {"pixel_box": (x1,y1,x2,y2), "class_code": str}
    pred_boxes: list of {"pixel_box": (x1,y1,x2,y2), "label": str}

    Returns overall tp/fp/fn plus a per-class breakdown.
    """
    gt_matched = [False] * len(gt_boxes)
    matches = []  # (iou, is_tp, gt_class_code or None, pred_label)
    per_class = {}  # class_code -> {"tp": int, "fp": int, "fn": int}

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
            # FP attributed to the class the model claimed it saw, even
            # though it didn't correctly match anything of that class.
            _bump(p["label"].strip().upper() or "UNKNOWN", "fp")

    # Any unmatched ground truth box is a false negative for its own class.
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


def build_experiment_config(raw_config: dict | None, model_type: str) -> dict:
    if not raw_config:
        return {"model_type": model_type}

    config = {
        "model_type": model_type,
        "device": raw_config.get("device", "cpu"),
        "dtype": str(raw_config.get("dtype", "float32")).replace("torch.", ""),
    }

    if model_type == "locateanything":
        config.update({
            "prompt": raw_config.get("prompt") or raw_config.get("prompt_used"),
            "min_box_area_fraction": raw_config.get("min_box_area_fraction"),
            "containment_threshold": raw_config.get("containment_threshold"),
            "max_new_tokens": raw_config.get("max_new_tokens"),
            "repetition_penalty": raw_config.get("repetition_penalty"),
            "no_repeat_ngram_size": raw_config.get("no_repeat_ngram_size"),
        })
    else:
        config["conf_threshold"] = raw_config.get("conf_threshold", 0.25)

    return config


def build_markdown_report(total_gt, total_pred, total_tp, total_fp, total_fn,
                          overall_precision, overall_recall, overall_f1, mean_iou,
                          per_image_metrics, experiment_config, model_type, per_class_summary=None):
    model_name = "YOLOv8" if model_type == "yolo" else "LocateAnything-3B"
    
    md = f"""# {model_name} Evaluation Report

**Model Type:** {model_type}  
**IoU Threshold:** {IOU_THRESHOLD}

## Experiment Configuration
"""

    md += f"- Model Type: **{model_type}**\n"
    md += f"- Device: {experiment_config.get('device')}\n"
    md += f"- Dtype: {experiment_config.get('dtype')}\n"

    if model_type == "locateanything":
        md += f"- Prompt: \"{experiment_config.get('prompt', 'N/A')}\"\n"
        md += f"- min_box_area_fraction: {experiment_config.get('min_box_area_fraction')}\n"
        md += f"- containment_threshold: {experiment_config.get('containment_threshold')}\n"
        md += f"- max_new_tokens: {experiment_config.get('max_new_tokens')}\n"
        md += f"- repetition_penalty: {experiment_config.get('repetition_penalty')}\n"

    md += f"""
## Overall Results

**Total Ground Truth boxes:** {total_gt}  
**Total Predictions:** {total_pred}

| Metric              | Value    |
|---------------------|----------|
| True Positives (TP) | {total_tp} |
| False Positives (FP)| {total_fp} |
| False Negatives (FN)| {total_fn} |
| **Precision**       | **{overall_precision:.4f}** |
| **Recall**          | **{overall_recall:.4f}** |
| **F1 Score**        | **{overall_f1:.4f}** |
| **Mean IoU**        | **{mean_iou:.4f}** |
"""

    if per_class_summary:
        md += """
## Per-Class Results

A true positive requires both IoU >= threshold AND the predicted label
matching the ground truth class — so this breakdown shows which species
the model actually confuses or misses, rather than just overlapping boxes
of the wrong class.

| Class | TP | FP | FN | Precision | Recall | F1 |
|-------|----|----|----|-----------|--------|-----|
"""
        for class_code in sorted(per_class_summary.keys()):
            c = per_class_summary[class_code]
            md += (f"| {class_code} | {c['tp']} | {c['fp']} | {c['fn']} | "
                   f"{c['precision']:.4f} | {c['recall']:.4f} | {c['f1']:.4f} |\n")

    md += """
## Per-Image Results

| Image | GT | Pred | TP | FP | FN | F1 Score |
|-------|----|------|----|----|----|----------|
"""

    for r in per_image_metrics:
        md += f"| {r['image'][:60]}... | {r['gt_count']} | {r['pred_count']} | {r['tp']} | {r['fp']} | {r['fn']} | {r['f1']:.3f} |\n"

    md += "\n---\n*Report generated by detection_evaluation.py*\n"
    return md


# ─── MAIN ────────────────────────────────────────────────────────────────────
def evaluate_model(detections_file: Path):
    model_type = "yolo" if "yolo" in detections_file.name else "locateanything"
    
    print(f"Evaluating {model_type.upper()} from: {detections_file.name}")

    with open(detections_file, "r", encoding="utf-8") as f:
        payload = json.load(f)

    raw_config = payload.get("config", {})
    all_results = payload.get("results", [])

    experiment_config = build_experiment_config(raw_config, model_type)

    total_tp = total_fp = total_fn = 0
    all_ious = []
    per_image_metrics = []
    overall_per_class = {}  # class_code -> {"tp":.., "fp":.., "fn":..}

    for res in all_results:
        img_name = res["image"]
        pred_dets = res.get("detections", [])

        image_path = IMAGES_DIR / img_name
        if not image_path.exists():
            for ext in [".jpg", ".png", ".jpeg"]:
                candidate = IMAGES_DIR / f"{Path(img_name).stem}{ext}"
                if candidate.exists():
                    image_path = candidate
                    break

        if image_path.exists():
            img_w, img_h = Image.open(image_path).size
        else:
            print(f"  ⚠️  Skipping {img_name}: not found in {IMAGES_DIR}")
            continue

        gt_norm = load_ground_truth(img_name)
        gt_boxes = [
            {"pixel_box": yolo_to_pixel(b, img_w, img_h), "class_code": b["class_code"]}
            for b in gt_norm
        ]
        pred_boxes = [
            {
                "pixel_box": (d["bbox"]["x1"], d["bbox"]["y1"], d["bbox"]["x2"], d["bbox"]["y2"]),
                "label": d.get("class", ""),
            }
            for d in pred_dets
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
            "image": img_name,
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

    # Finalize per-class precision/recall/F1
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

    # Overall metrics
    total_gt = total_tp + total_fn
    total_pred = total_tp + total_fp
    precision = total_tp / total_pred if total_pred > 0 else 0.0
    recall = total_tp / total_gt if total_gt > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    mean_iou = np.mean(all_ious) if all_ious else 0.0

    base_name = "yolo" if model_type == "yolo" else "locateanything"
    json_file = RESULTS_ROOT / f"{base_name}_evaluation_results.json"
    md_file = RESULTS_ROOT / f"{base_name}_evaluation_results.md"

    with open(json_file, "w", encoding="utf-8") as f:
        json.dump({
            "experiment_config": experiment_config,
            "overall": {
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "mean_iou": mean_iou,
                "tp": total_tp,
                "fp": total_fp,
                "fn": total_fn,
                "total_gt": total_gt,
                "total_pred": total_pred
            },
            "per_class": per_class_summary,
            "per_image": per_image_metrics
        }, f, indent=2)

    report_text = build_markdown_report(
        total_gt, total_pred, total_tp, total_fp, total_fn,
        precision, recall, f1, mean_iou, per_image_metrics, experiment_config, model_type,
        per_class_summary=per_class_summary
    )

    with open(md_file, "w", encoding="utf-8") as f:
        f.write(report_text)

    print(f"   → JSON: {json_file.name}")
    print(f"   → MD:   {md_file.name}")


# ─── MAIN ────────────────────────────────────────────────────────────────────
def main():
    print("=" * 80)
    print("Starting Evaluation for All Available Models")
    print("=" * 80)

    evaluated = False

    for file_name in ["yolo_detections.json", "locateanything_detections.json"]:
        detections_file = RESULTS_ROOT / file_name
        if detections_file.exists():
            evaluate_model(detections_file)
            evaluated = True

    if not evaluated:
        print(f"❌ No detections files found in {RESULTS_ROOT}")
        print("Please run weed_detection.py first.")
        sys.exit(1)

    print("\n✅ All available evaluations completed!")


if __name__ == "__main__":
    main()