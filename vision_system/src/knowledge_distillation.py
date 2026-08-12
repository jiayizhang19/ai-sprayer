"""
Knowledge Distillation: YOLO26m (teacher) → YOLO26s (student)
-------------------------------------------------------------
Trains a YOLO26s student under guidance of your trained YOLO26m.
Results are saved under runs/student_models/.

The final markdown report compares three models side-by-side:
  1. YOLO26m          (teacher)
  2. YOLO26s baseline (no KD)
  3. YOLO26s distilled

Metrics included:
  - Overall accuracy (Precision / Recall / mAP50 / mAP50-95)
  - Model size (parameters + file size on disk)
  - Class-wise performance of the distilled student

pip install -U ultralytics
"""

import sys
import os
from pathlib import Path
from datetime import datetime

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRAIN_CONFIG_PATH = PROJECT_ROOT / "config" / "train_config.yaml"

# ──────────────────────────────────────────────────────────────
# Edit these paths if your run folder names are different
# ──────────────────────────────────────────────────────────────
TEACHER_WEIGHTS    = PROJECT_ROOT / "runs/train/yolo26m_ep150_b8_lr0.001/weights/best.pt"
BASELINE_S_WEIGHTS = PROJECT_ROOT / "runs/train/yolo26s_ep150_b8_lr0.001/weights/best.pt"
STUDENT_SIZE       = "s"          # "s" or "n"
PROJECT_DIR        = PROJECT_ROOT / "runs" / "student_models"


def load_train_config(path: Path = TRAIN_CONFIG_PATH) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    cfg = {}
    cfg["DATA_YAML"] = PROJECT_ROOT / raw["dataset"]["data_yaml"]
    cfg["BASE_WEIGHTS"] = f"yolo26{STUDENT_SIZE}.pt"
    cfg["PROJECT_DIR"] = PROJECT_DIR
    cfg["EPOCHS"] = raw["training"]["epochs"]
    cfg["IMAGE_SIZE"] = raw["training"]["image_size"]
    cfg["BATCH_SIZE"] = raw["training"]["batch_size"]
    cfg["DEVICE"] = raw["training"]["device"]
    cfg["OPTIMIZER"] = raw["training"].get("optimizer", "auto")
    cfg["LR0"] = raw["training"].get("lr0")
    cfg["MOMENTUM"] = raw["training"].get("momentum")
    cfg["PATIENCE"] = raw["training"]["patience"]
    cfg["WORKERS"] = raw["training"]["workers"]
    cfg["RESUME"] = raw["training"]["resume"]

    aug = raw["augmentation"]
    cfg["HSV_H"] = aug["hsv_h"]
    cfg["HSV_S"] = aug["hsv_s"]
    cfg["HSV_V"] = aug["hsv_v"]
    cfg["DEGREES"] = aug["degrees"]
    cfg["TRANSLATE"] = aug["translate"]
    cfg["SCALE"] = aug["scale"]
    cfg["FLIPLR"] = aug["fliplr"]
    cfg["FLIPUD"] = aug["flipud"]
    cfg["MOSAIC"] = aug["mosaic"]

    cfg["VAL_IOU"] = raw["validation"]["iou_threshold"]
    cfg["VAL_CONF"] = raw["validation"]["conf_threshold"]
    return cfg


def build_run_name(cfg: dict) -> str:
    model_version = Path(cfg["BASE_WEIGHTS"]).stem
    lr0 = cfg["LR0"]
    lr_tag = f"lr{lr0}" if lr0 is not None else "lrauto"
    return (
        f"{model_version}"
        f"_ep{cfg['EPOCHS']}"
        f"_b{cfg['BATCH_SIZE']}"
        f"_{lr_tag}"
        f"_distill"
    )


def build_runtime_data_yaml(data_yaml_path: Path) -> Path:
    with open(data_yaml_path, "r", encoding="utf-8") as f:
        data_cfg = yaml.safe_load(f)

    dataset_root = (data_yaml_path.parent / data_cfg["path"]).resolve()
    runtime_cfg = dict(data_cfg)
    runtime_cfg["path"] = str(dataset_root)

    runtime_yaml = PROJECT_ROOT / "config" / "data_runtime.yaml"
    with open(runtime_yaml, "w", encoding="utf-8") as f:
        yaml.safe_dump(runtime_cfg, f, sort_keys=False)
    return runtime_yaml


def run_validation(model, runtime_data_yaml, cfg, run_dir, tag: str):
    metrics = model.val(
        data=str(runtime_data_yaml),
        iou=cfg["VAL_IOU"],
        conf=cfg["VAL_CONF"],
        project=str(run_dir.parent),
        name=f"{run_dir.name}/val_{tag}",
        exist_ok=True,
        verbose=True,
        plots=True,
    )
    return metrics


def _extract(m):
    if m is None:
        return None
    return {
        "precision": float(m.box.mp),
        "recall": float(m.box.mr),
        "map50": float(m.box.map50),
        "map": float(m.box.map),
        "names": m.names,
        "classes": getattr(m, "classes", None),
        "p": m.box.p,
        "r": m.box.r,
        "ap50": m.box.ap50,
        "ap": m.box.ap,
    }


def get_model_stats(weights_path: Path):
    """Return file size (MB) and parameter count (millions)."""
    if weights_path is None or not Path(weights_path).exists():
        return None

    from ultralytics import YOLO

    path = Path(weights_path)
    file_size_mb = os.path.getsize(path) / (1024 * 1024)

    model = YOLO(str(path))
    params = sum(p.numel() for p in model.model.parameters())
    params_m = params / 1e6

    return {
        "file_size_mb": file_size_mb,
        "params_m": params_m,
        "path": path,
    }


def write_three_way_report(
    run_dir: Path,
    teacher_metrics,
    baseline_s_metrics,
    distilled_metrics,
    teacher_path: Path,
    baseline_s_path: Path,
    distilled_path: Path,
):
    """Write markdown report with model size + accuracy comparison."""
    report_path = run_dir / "summaried_performance.md"

    t = _extract(teacher_metrics)
    b = _extract(baseline_s_metrics)
    d = _extract(distilled_metrics)

    t_stats = get_model_stats(teacher_path)
    b_stats = get_model_stats(baseline_s_path)
    d_stats = get_model_stats(distilled_path)

    def fmt(v, decimals=4):
        return f"{v:.{decimals}f}" if v is not None else "—"

    def delta(new, old):
        if new is None or old is None:
            return "—"
        diff = new - old
        sign = "+" if diff >= 0 else ""
        return f"{sign}{diff:.4f}"

    md = f"""# Knowledge Distillation Performance Summary
**Date:** {datetime.now().strftime("%Y-%m-%d %H:%M")}  
**Teacher:** `{teacher_path.name}`  
**Student baseline:** `{baseline_s_path.name if baseline_s_path else "N/A"}`  
**Student distilled:** `{run_dir.name}`  

---

## 1. Model Size Comparison

| Model | Parameters (M) | File Size (MB) | Notes |
|:------|:--------------:|:--------------:|:------|
| **YOLO26m (Teacher)** | {fmt(t_stats['params_m'], 2) if t_stats else "—"} | {fmt(t_stats['file_size_mb'], 1) if t_stats else "—"} | Upper-bound accuracy |
| **YOLO26s (Baseline)** | {fmt(b_stats['params_m'], 2) if b_stats else "—"} | {fmt(b_stats['file_size_mb'], 1) if b_stats else "—"} | Same architecture, no KD |
| **YOLO26s (Distilled)** | {fmt(d_stats['params_m'], 2) if d_stats else "—"} | {fmt(d_stats['file_size_mb'], 1) if d_stats else "—"} | Same architecture + KD |

> Distilled and baseline YOLO26s should have **identical** parameter count and nearly identical file size.

---

## 2. Overall Accuracy (Validation Split)

| Metric | YOLO26m<br>(Teacher) | YOLO26s<br>(Baseline) | YOLO26s<br>(Distilled) | Δ vs Baseline | Δ vs Teacher |
|:-------|:--------------------:|:---------------------:|:----------------------:|:-------------:|:------------:|
| **Precision** | {fmt(t and t['precision'])} | {fmt(b and b['precision'])} | **{fmt(d and d['precision'])}** | {delta(d and d['precision'], b and b['precision'])} | {delta(d and d['precision'], t and t['precision'])} |
| **Recall**    | {fmt(t and t['recall'])}    | {fmt(b and b['recall'])}    | **{fmt(d and d['recall'])}**    | {delta(d and d['recall'], b and b['recall'])}       | {delta(d and d['recall'], t and t['recall'])} |
| **mAP50**     | {fmt(t and t['map50'])}     | {fmt(b and b['map50'])}     | **{fmt(d and d['map50'])}**     | {delta(d and d['map50'], b and b['map50'])}         | {delta(d and d['map50'], t and t['map50'])} |
| **mAP50-95**  | {fmt(t and t['map'])}       | {fmt(b and b['map'])}       | **{fmt(d and d['map'])}**       | {delta(d and d['map'], b and b['map'])}             | {delta(d and d['map'], t and t['map'])} |

> Positive Δ means the distilled student improved over that reference.

---

## 3. Class-Specific Performance (Distilled Student)

| Class ID | Class Name | Precision | Recall | mAP50 | mAP50-95 |
|:--------:|:-----------|:---------:|:------:|:-----:|:--------:|
"""
    if d and d["classes"] is not None:
        for idx, class_index in enumerate(d["classes"]):
            name = d["names"][class_index]
            md += (
                f"| {class_index} | **{name}** | "
                f"{d['p'][idx]:.4f} | {d['r'][idx]:.4f} | "
                f"{d['ap50'][idx]:.4f} | {d['ap'][idx]:.4f} |\n"
            )
    else:
        md += "| — | (class metrics not available) | — | — | — | — |\n"

    md += f"""
---

## 4. Model Paths

| Role | Path |
|:-----|:-----|
| Teacher (YOLO26m) | `{teacher_path}` |
| Baseline (YOLO26s) | `{baseline_s_path}` |
| Distilled (YOLO26s) | `{distilled_path}` |

## Notes
- Teacher is frozen during training; only the student + projector are updated.
- Final `best.pt` contains **only the student** → identical parameter count and inference cost to a normal YOLO26s.
- Distillation loss weight used: `dis=6.0` (increase to 8–10 for stronger teacher influence).
"""

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"📝 Three-way report (size + accuracy) saved to: {report_path}")
    return report_path


def train():
    from ultralytics import YOLO

    cfg = load_train_config()
    cfg["RUN_NAME"] = build_run_name(cfg)

    print("=" * 70)
    print("Knowledge Distillation — YOLO26m → YOLO26s")
    print("=" * 70)

    for p, label in [
        (TEACHER_WEIGHTS, "Teacher (YOLO26m)"),
        (BASELINE_S_WEIGHTS, "Baseline YOLO26s"),
    ]:
        if not p.exists():
            print(f"⚠️  {label} not found:\n     {p}")
            print("     Update the path at the top of this script if the run name differs.")
        else:
            print(f"✓ {label}: {p}")

    runtime_data_yaml = build_runtime_data_yaml(cfg["DATA_YAML"])

    print(f"\nStudent start weights : {cfg['BASE_WEIGHTS']}")
    print(f"Device                : {cfg['DEVICE']}")
    print(f"Epochs / Batch / imgsz: {cfg['EPOCHS']} / {cfg['BATCH_SIZE']} / {cfg['IMAGE_SIZE']}")
    print(f"Output run            : {cfg['PROJECT_DIR'] / cfg['RUN_NAME']}")
    print("-" * 70)

    student = YOLO(cfg["BASE_WEIGHTS"])

    try:
        # ── Train with knowledge distillation ───────────────────
        student.train(
            data=str(runtime_data_yaml),
            epochs=cfg["EPOCHS"],
            imgsz=cfg["IMAGE_SIZE"],
            batch=cfg["BATCH_SIZE"],
            device=cfg["DEVICE"],
            optimizer=cfg["OPTIMIZER"],
            lr0=cfg["LR0"],
            momentum=cfg["MOMENTUM"],
            patience=cfg["PATIENCE"],
            workers=cfg["WORKERS"],
            resume=cfg["RESUME"],
            project=str(cfg["PROJECT_DIR"]),
            name=cfg["RUN_NAME"],

            distill_model=str(TEACHER_WEIGHTS),
            dis=6.0,

            hsv_h=cfg["HSV_H"],
            hsv_s=cfg["HSV_S"],
            hsv_v=cfg["HSV_V"],
            degrees=cfg["DEGREES"],
            translate=cfg["TRANSLATE"],
            scale=cfg["SCALE"],
            fliplr=cfg["FLIPLR"],
            flipud=cfg["FLIPUD"],
            mosaic=cfg["MOSAIC"],

            exist_ok=True,
            plots=True,
            verbose=True,
        )

        run_dir = cfg["PROJECT_DIR"] / cfg["RUN_NAME"]
        best_weights = run_dir / "weights" / "best.pt"
        print(f"\nBest distilled student weights: {best_weights}")

        # ── Validate all three models ───────────────────────────
        print("\n[1/3] Validating distilled YOLO26s ...")
        distilled_metrics = run_validation(
            student, runtime_data_yaml, cfg, run_dir, tag="distilled"
        )

        teacher_metrics = None
        if TEACHER_WEIGHTS.exists():
            print("\n[2/3] Validating teacher YOLO26m ...")
            teacher_model = YOLO(str(TEACHER_WEIGHTS))
            teacher_metrics = run_validation(
                teacher_model, runtime_data_yaml, cfg, run_dir, tag="teacher_m"
            )

        baseline_s_metrics = None
        if BASELINE_S_WEIGHTS.exists():
            print("\n[3/3] Validating baseline YOLO26s ...")
            baseline_model = YOLO(str(BASELINE_S_WEIGHTS))
            baseline_s_metrics = run_validation(
                baseline_model, runtime_data_yaml, cfg, run_dir, tag="baseline_s"
            )
        else:
            print("\n⚠️  Baseline YOLO26s weights not found — baseline column will show '—'.")

        # ── Write the complete markdown report ──────────────────
        write_three_way_report(
            run_dir=run_dir,
            teacher_metrics=teacher_metrics,
            baseline_s_metrics=baseline_s_metrics,
            distilled_metrics=distilled_metrics,
            teacher_path=TEACHER_WEIGHTS,
            baseline_s_path=BASELINE_S_WEIGHTS if BASELINE_S_WEIGHTS.exists() else None,
            distilled_path=best_weights,
        )

        print("\n" + "=" * 70)
        print("Distillation + three-way comparison finished.")
        print(f"Report → {run_dir / 'summaried_performance.md'}")
        print("=" * 70)

    finally:
        if runtime_data_yaml.exists():
            try:
                runtime_data_yaml.unlink()
            except Exception:
                pass


if __name__ == "__main__":
    train()