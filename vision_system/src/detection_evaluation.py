"""
Evaluation Script: LocateAnything vs Ground Truth
-------------------------------------------------
Computes standard object detection metrics:
- Precision, Recall, F1-score
- mIoU (mean Intersection over Union)
- Average IoU of matched detections
- TP/FP/FN counts
- Option to use different IoU thresholds

Run from vision_system/:
    python evaluate_detection.py
"""

import json
from pathlib import Path
import sys
from collections import defaultdict

import numpy as np
from PIL import Image

# ─── CONFIG ──────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.config_loader import load_config

GT_LABELS_DIR = PROJECT_ROOT / "labels"
DETECTIONS_JSON = PROJECT_ROOT / "detections.json"   # from weed_detection_step1.py

IOU_THRESHOLD = 0.5          # Standard threshold for matching
MIN_CONF = 0.0               # Not used currently (model has no confidence)

print(f"Using IoU threshold = {IOU_THRESHOLD}\n")


# ─── HELPERS ─────────────────────────────────────────────────────────────────

def load_ground_truth(image_name: str) -> list[dict]:
    """Load YOLO .txt ground truth for an image."""
    stem = Path(image_name).stem.replace("detected_", "").replace("gt_", "")
    label_path = GT_LABELS_DIR / f"{stem}.txt"
    
    if not label_path.exists():
        # Try with original capture name variations
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
                "x_center": x_c, "y_center": y_c,
                "width": w, "height": h
            })
    return boxes


def yolo_to_pixel(box: dict, img_w: int, img_h: int) -> tuple[int, int, int, int]:
    """Convert normalized YOLO box to pixel (x1, y1, x2, y2)."""
    cx = box["x_center"] * img_w
    cy = box["y_center"] * img_h
    w = box["width"] * img_w
    h = box["height"] * img_h
    x1 = int(round(cx - w / 2))
    y1 = int(round(cy - h / 2))
    x2 = int(round(cx + w / 2))
    y2 = int(round(cy + h / 2))
    return max(0, x1), max(0, y1), min(img_w, x2), min(img_h, y2)


def compute_iou(box1: tuple, box2: tuple) -> float:
    """Compute IoU between two boxes (x1,y1,x2,y2)."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    if x2 < x1 or y2 < y1:
        return 0.0

    inter_area = (x2 - x1) * (y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    if area1 == 0 or area2 == 0:
        return 0.0

    union_area = area1 + area2 - inter_area
    return inter_area / union_area


def match_detections(gt_boxes: list, pred_boxes: list, iou_thresh: float = 0.5):
    """Greedy matching of predictions to ground truth."""
    gt_matched = [False] * len(gt_boxes)
    matches = []

    for p in pred_boxes:
        best_iou = 0.0
        best_idx = -1
        for i, gt in enumerate(gt_boxes):
            if gt_matched[i]:
                continue
            iou = compute_iou(p, gt)
            if iou > best_iou:
                best_iou = iou
                best_idx = i

        if best_iou >= iou_thresh:
            gt_matched[best_idx] = True
            matches.append((best_iou, True))  # TP
        else:
            matches.append((0.0, False))      # FP

    tp = sum(1 for _, is_tp in matches if is_tp)
    fp = len(pred_boxes) - tp
    fn = len(gt_boxes) - tp

    ious = [iou for iou, _ in matches if iou > 0]

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "matched_ious": ious,
        "precision": tp / (tp + fp) if (tp + fp) > 0 else 0.0,
        "recall": tp / (tp + fn) if (tp + fn) > 0 else 0.0,
    }


def normalize_detection_payload(payload):
    """Support both legacy list-only detections.json and the new metadata wrapper."""
    if isinstance(payload, dict):
        return payload.get("experiment_config", {}), payload.get("results", [])

    return {}, payload


def build_experiment_config(raw_config: dict | None) -> dict:
    """Build a serializable experiment config for reports."""
    if not raw_config:
        return {}

    return {
        "model_id": raw_config.get("MODEL_ID"),
        "device": raw_config.get("DEVICE"),
        "dtype": str(raw_config.get("DTYPE")).replace("torch.", ""),
        "prompt_used": raw_config.get("DETECTION_PROMPTS", [None])[0],
        "prompt": raw_config.get("DETECTION_PROMPTS", [None])[0],
        "min_box_area_fraction": raw_config.get("MIN_BOX_AREA_FRACTION"),
        "containment_threshold": raw_config.get("CONTAINMENT_THRESHOLD"),
        "conf_threshold": raw_config.get("CONF_THRESHOLD"),
        "max_new_tokens": raw_config.get("MAX_NEW_TOKENS"),
        "repetition_penalty": raw_config.get("REPETITION_PENALTY"),
        "no_repeat_ngram_size": raw_config.get("NO_REPEAT_NGRAM_SIZE"),
    }


def build_markdown_report(total_gt: int, total_pred: int, total_tp: int, total_fp: int,
                          total_fn: int, overall_precision: float, overall_recall: float,
                          overall_f1: float, mean_iou: float, all_ious: list[float],
                          per_image_metrics: list[dict], experiment_config: dict) -> str:
    """Build a markdown evaluation report that mirrors the terminal summary."""
    lines = [
        "# LocateAnything vs Ground Truth Evaluation",
        "",
        "## Experiment Config",
        "",
    ]

    if experiment_config:
        lines.extend([
            f"- Model ID: {experiment_config.get('model_id', 'n/a')}",
            f"- Device: {experiment_config.get('device', 'n/a')}",
            f"- Dtype: {experiment_config.get('dtype', 'n/a')}",
            f"- Prompt used: {experiment_config.get('prompt_used', experiment_config.get('prompt', 'n/a'))}",
            f"- min_box_area_fraction: {experiment_config.get('min_box_area_fraction', 'n/a')}",
            f"- containment_threshold: {experiment_config.get('containment_threshold', 'n/a')}",
            f"- conf_threshold: {experiment_config.get('conf_threshold', 'n/a')}",
            f"- max_new_tokens: {experiment_config.get('max_new_tokens', 'n/a')}",
            f"- repetition_penalty: {experiment_config.get('repetition_penalty', 'n/a')}",
            f"- no_repeat_ngram_size: {experiment_config.get('no_repeat_ngram_size', 'n/a')}",
            "",
        ])
    else:
        lines.extend([
            "- No experiment metadata was found in detections.json.",
            "",
        ])

    lines.extend([
        f"- IoU threshold: {IOU_THRESHOLD}",
        f"- Total ground truth boxes: {total_gt}",
        f"- Total predictions: {total_pred}",
        f"- True positives (TP): {total_tp}",
        f"- False positives (FP): {total_fp}",
        f"- False negatives (FN): {total_fn}",
        f"- Precision: {overall_precision:.4f}",
        f"- Recall: {overall_recall:.4f}",
        f"- F1 score: {overall_f1:.4f}",
        f"- Mean IoU (matched): {mean_iou:.4f}",
        f"- Total matched boxes: {len(all_ious)}",
        "",
        "## Per-image results",
        "",
        "| Image | GT | Pred | TP | FP | FN | Precision | Recall | F1 | Mean IoU |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])

    for metrics in per_image_metrics:
        lines.append(
            f"| {metrics['image']} | {metrics['gt_count']} | {metrics['pred_count']} | "
            f"{metrics['tp']} | {metrics['fp']} | {metrics['fn']} | "
            f"{metrics['precision']:.4f} | {metrics['recall']:.4f} | {metrics['f1']:.4f} | "
            f"{metrics['mean_iou_matched']:.4f} |"
        )

    lines.extend([
        "",
        "## Notes",
        "- Ground truth boxes are loaded from YOLO `.txt` files in `labels/`.",
        "- Predictions are loaded from `detections.json` produced by the detection script.",
        "- IoU matching uses a greedy one-to-one assignment at the threshold above.",
    ])

    return "\n".join(lines)


# ─── MAIN ────────────────────────────────────────────────────────────────────

def main():
    if not DETECTIONS_JSON.exists():
        print(f"❌ Detections file not found: {DETECTIONS_JSON}")
        print("Please run weed_detection_step1.py first.")
        sys.exit(1)

    with open(DETECTIONS_JSON, "r", encoding="utf-8") as f:
        detection_payload = json.load(f)

    raw_experiment_config, all_results = normalize_detection_payload(detection_payload)
    if not raw_experiment_config:
        raw_experiment_config = build_experiment_config(load_config())
    else:
        raw_experiment_config = {
            "model_id": raw_experiment_config.get("model_id"),
            "device": raw_experiment_config.get("device"),
            "dtype": raw_experiment_config.get("dtype"),
            "prompt": raw_experiment_config.get("prompt"),
            "min_box_area_fraction": raw_experiment_config.get("min_box_area_fraction"),
            "containment_threshold": raw_experiment_config.get("containment_threshold"),
            "conf_threshold": raw_experiment_config.get("conf_threshold"),
            "max_new_tokens": raw_experiment_config.get("max_new_tokens"),
            "repetition_penalty": raw_experiment_config.get("repetition_penalty"),
            "no_repeat_ngram_size": raw_experiment_config.get("no_repeat_ngram_size"),
        }

    total_tp = total_fp = total_fn = 0
    all_ious = []
    per_image_metrics = []

    print("=" * 80)
    print("Evaluating LocateAnything-3B vs Ground Truth")
    print("=" * 80)
    if raw_experiment_config:
        print(f"Prompt used              : {raw_experiment_config.get('prompt_used', raw_experiment_config.get('prompt', 'n/a'))}")
        print(f"Min box area fraction    : {raw_experiment_config.get('min_box_area_fraction', 'n/a')}")
        print(f"Containment threshold    : {raw_experiment_config.get('containment_threshold', 'n/a')}")

    for res in all_results:
        img_name = res["image"]
        pred_dets = res.get("detections", [])

        # Get image dimensions
        image_path = PROJECT_ROOT / "weed_images" / img_name
        if not image_path.exists():
            # Try to find original image
            for ext in [".jpg", ".png", ".jpeg"]:
                candidate = PROJECT_ROOT / "weed_images" / f"{Path(img_name).stem}{ext}"
                if candidate.exists():
                    image_path = candidate
                    break

        if image_path.exists():
            pil_img = Image.open(image_path)
            img_w, img_h = pil_img.size
        else:
            print(f"⚠️  Could not find image for {img_name}, skipping dimensions")
            continue

        # Load GT
        gt_norm_boxes = load_ground_truth(img_name)
        gt_pixel_boxes = [yolo_to_pixel(b, img_w, img_h) for b in gt_norm_boxes]

        # Convert predictions to pixel boxes
        pred_pixel_boxes = []
        for d in pred_dets:
            pred_pixel_boxes.append((d["x1"], d["y1"], d["x2"], d["y2"]))

        # Match
        metrics = match_detections(gt_pixel_boxes, pred_pixel_boxes, IOU_THRESHOLD)

        total_tp += metrics["tp"]
        total_fp += metrics["fp"]
        total_fn += metrics["fn"]
        all_ious.extend(metrics["matched_ious"])

        image_f1 = 2 * metrics["precision"] * metrics["recall"] / (metrics["precision"] + metrics["recall"] + 1e-8)

        per_image_metrics.append({
            "image": img_name,
            "gt_count": len(gt_pixel_boxes),
            "pred_count": len(pred_pixel_boxes),
            "tp": metrics["tp"],
            "fp": metrics["fp"],
            "fn": metrics["fn"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": image_f1,
            "mean_iou_matched": np.mean(metrics["matched_ious"]) if metrics["matched_ious"] else 0.0
        })

        print(f"{img_name:40}  GT: {len(gt_pixel_boxes):2d} | Pred: {len(pred_pixel_boxes):2d} | "
              f"TP: {metrics['tp']:2d} FP: {metrics['fp']:2d} FN: {metrics['fn']:2d} | "
              f"F1: {image_f1:.3f}")

    # ─── Overall Metrics ─────────────────────────────────────────────────────
    total_gt = total_tp + total_fn
    total_pred = total_tp + total_fp

    overall_precision = total_tp / total_pred if total_pred > 0 else 0.0
    overall_recall = total_tp / total_gt if total_gt > 0 else 0.0
    overall_f1 = 2 * overall_precision * overall_recall / (overall_precision + overall_recall + 1e-8)
    mean_iou = np.mean(all_ious) if all_ious else 0.0

    print("\n" + "=" * 80)
    print("OVERALL RESULTS")
    print("=" * 80)
    print(f"Total Ground Truth boxes : {total_gt}")
    print(f"Total Predictions        : {total_pred}")
    print(f"True Positives (TP)      : {total_tp}")
    print(f"False Positives (FP)     : {total_fp}")
    print(f"False Negatives (FN)     : {total_fn}")
    print("-" * 60)
    print(f"Precision                : {overall_precision:.4f}")
    print(f"Recall                   : {overall_recall:.4f}")
    print(f"F1 Score                 : {overall_f1:.4f}")
    print(f"Mean IoU (matched)       : {mean_iou:.4f}")
    print(f"Total matched boxes      : {len(all_ious)}")

    # Save detailed results
    output_file = PROJECT_ROOT / "evaluation_results.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump({
            "experiment_config": raw_experiment_config,
            "overall": {
                "precision": overall_precision,
                "recall": overall_recall,
                "f1": overall_f1,
                "mean_iou": mean_iou,
                "tp": total_tp,
                "fp": total_fp,
                "fn": total_fn,
                "total_gt": total_gt,
                "total_pred": total_pred
            },
            "per_image": per_image_metrics
        }, f, indent=2)

    report_file = PROJECT_ROOT / "evaluation_results.md"
    report_text = build_markdown_report(
        total_gt,
        total_pred,
        total_tp,
        total_fp,
        total_fn,
        overall_precision,
        overall_recall,
        overall_f1,
        mean_iou,
        all_ious,
        per_image_metrics,
        raw_experiment_config,
    )
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(report_text + "\n")

    print(f"\nDetailed results saved to: {output_file.name}")
    print(f"Markdown report saved to:   {report_file.name}")


if __name__ == "__main__":
    main()