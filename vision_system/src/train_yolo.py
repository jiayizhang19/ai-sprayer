"""
YOLOv8 Fine-Tuning on the OPPD Weed Dataset
--------------------------------------------
Fine-tunes a COCO-pretrained YOLOv8 checkpoint on your labeled weed dataset
and saves a detailed markdown performance summary directly to the run directory.
"""

import sys
import shutil
from pathlib import Path

import yaml

# This script lives in vision_system/src/. config/ lives in vision_system/.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRAIN_CONFIG_PATH = PROJECT_ROOT / "config" / "train_config.yaml"


def load_train_config(path: Path = TRAIN_CONFIG_PATH) -> dict:
    """Load train_config.yaml and resolve relative paths against project root."""
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    cfg = {}

    # --- dataset ---
    cfg["DATA_YAML"] = PROJECT_ROOT / raw["dataset"]["data_yaml"]

    # --- model ---
    cfg["BASE_WEIGHTS"] = raw["model"]["base_weights"]
    cfg["PROJECT_DIR"] = PROJECT_ROOT / raw["model"]["project_dir"]

    # --- training ---
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

    # --- augmentation ---
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

    # --- validation ---
    cfg["VAL_IOU"] = raw["validation"]["iou_threshold"]
    cfg["VAL_CONF"] = raw["validation"]["conf_threshold"]

    return cfg


def build_run_name(cfg: dict) -> str:
    """Auto-generate a unique run name from the actual hyperparameters used."""
    model_version = Path(cfg["BASE_WEIGHTS"]).stem
    lr0 = cfg["LR0"]
    lr_tag = f"lr{lr0}" if lr0 is not None else "lrauto"

    return (
        f"{model_version}"
        f"_ep{cfg['EPOCHS']}"
        f"_b{cfg['BATCH_SIZE']}"
        f"_{lr_tag}"
    )


def verify_dataset(data_yaml_path: Path) -> dict:
    """Basic sanity check before kicking off a training run."""
    if not data_yaml_path.exists():
        raise FileNotFoundError(
            f"Dataset YAML not found: {data_yaml_path}\n"
            f"Check 'data_yaml' in config/train_config.yaml."
        )

    with open(data_yaml_path, "r", encoding="utf-8") as f:
        data_cfg = yaml.safe_load(f)

    dataset_root = (data_yaml_path.parent / data_cfg["path"]).resolve()

    print(f"Dataset root: {dataset_root}")
    print(f"Classes: {len(data_cfg['names'])}")

    def _find_label_source(label_root: Path, stem: str) -> Path | None:
        direct = label_root / f"{stem}.txt"
        if direct.exists():
            return direct
        for candidate in label_root.glob("*.txt"):
            if candidate.stem == stem:
                return candidate
        return None

    flat_label_root = dataset_root / "labels"
    if flat_label_root.exists():
        for split in ("train", "val", "test"):
            split_rel = data_cfg.get(split)
            if not split_rel:
                continue
            img_dir = (dataset_root / split_rel).resolve()
            split_label_dir = flat_label_root / split
            split_label_dir.mkdir(parents=True, exist_ok=True)

            for img_file in img_dir.iterdir() if img_dir.exists() else []:
                if img_file.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
                    continue
                source_label = _find_label_source(flat_label_root, img_file.stem)
                if source_label is None:
                    continue
                target_label = split_label_dir / source_label.name
                if not target_label.exists():
                    shutil.copy2(source_label, target_label)

    for split in ("train", "val"):
        split_rel = data_cfg.get(split)
        if not split_rel:
            continue
        img_dir = (dataset_root / split_rel).resolve()
        if not img_dir.exists():
            raise FileNotFoundError(
                f"'{split}' image folder not found: {img_dir}\n"
                f"Check your dataset folder layout and 'path' in {data_yaml_path.name}."
            )
        img_count = sum(
            1 for f in img_dir.iterdir()
            if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        )
        if img_count == 0:
            raise ValueError(f"'{split}' image folder is empty: {img_dir}")
        print(f"  {split}: {img_count} images found in {img_dir}")

        label_dir = Path(str(img_dir).replace("images", "labels", 1))
        if not label_dir.exists():
            print(f"  ⚠️  WARNING: expected labels folder not found: {label_dir}")
        else:
            label_count = sum(1 for f in label_dir.iterdir() if f.suffix == ".txt")
            print(f"  {split}: {label_count} label files found in {label_dir}")

    return data_cfg


def build_runtime_data_yaml(data_yaml_path: Path) -> Path:
    """Create a persistent data runtime configuration inside the config directory to prevent OS sweeps."""
    with open(data_yaml_path, "r", encoding="utf-8") as f:
        data_cfg = yaml.safe_load(f)

    dataset_root = (data_yaml_path.parent / data_cfg["path"]).resolve()
    runtime_cfg = dict(data_cfg)
    runtime_cfg["path"] = str(dataset_root)

    # Save to standard config folder so it persists safely throughout the entire pipeline lifetime
    runtime_yaml = PROJECT_ROOT / "config" / "data_runtime.yaml"
    with open(runtime_yaml, "w", encoding="utf-8") as f:
        yaml.safe_dump(runtime_cfg, f, sort_keys=False)

    return runtime_yaml


def train():
    from ultralytics import YOLO

    cfg = load_train_config()
    cfg["RUN_NAME"] = build_run_name(cfg)

    print("=" * 70)
    print("YOLOv8 Fine-Tuning — OPPD Weed Dataset")
    print("=" * 70)

    print("\nVerifying dataset...")
    verify_dataset(cfg["DATA_YAML"])
    runtime_data_yaml = build_runtime_data_yaml(cfg["DATA_YAML"])

    print(f"\nBase weights: {cfg['BASE_WEIGHTS']}")
    print(f"Device: {cfg['DEVICE']}")
    print(f"Optimizer: {cfg['OPTIMIZER']} | lr0: {cfg['LR0']} | momentum: {cfg['MOMENTUM']}")
    print(f"Epochs: {cfg['EPOCHS']} | Batch size: {cfg['BATCH_SIZE']} | "
          f"Image size: {cfg['IMAGE_SIZE']}")
    print(f"Run name: {cfg['RUN_NAME']}")
    print(f"Output dir: {cfg['PROJECT_DIR']}")
    print("-" * 70)

    model = YOLO(cfg["BASE_WEIGHTS"])

    try:
        model.train(
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

            # Augmentation
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

        print("\n" + "=" * 70)
        print("Training complete.")

        run_dir = cfg["PROJECT_DIR"] / cfg["RUN_NAME"]
        best_weights = run_dir / "weights" / "best.pt"
        print(f"Best weights saved to: {best_weights}")
        print("\nTo use this model in your inference pipeline, set in config.yaml:")
        print(f'  yolo_model: "{best_weights}"')

        print("\nRunning final validation...")
        metrics = model.val(
            data=str(runtime_data_yaml),
            iou=cfg["VAL_IOU"],
            conf=cfg["VAL_CONF"],
            project=str(cfg["PROJECT_DIR"]),
            name=f"{cfg['RUN_NAME']}/val",
            exist_ok=True,
            verbose=True,
            plots=True
        )
        
        print(f"\nmAP50:    {metrics.box.map50:.4f}")
        print(f"mAP50-95: {metrics.box.map:.4f}")
        print(f"Precision: {metrics.box.mp:.4f}")
        print(f"Recall:    {metrics.box.mr:.4f}")

        # ─── EXPORT SUMMARY REPORT TO .MD ────────────────────────────────────
        report_target_path = run_dir / "summaried_performance.md"
        
        md_content = f"""# YOLOv8 Training Run Performance Summary
**Run Target Directory:** `{cfg['RUN_NAME']}`  
**Model Weight Checkpoint:** `{best_weights.name}`

## Overall Performance Summary Split Metrics
| Metric Name | Value |
| :--- | :---: |
| **Precision** | {metrics.box.mp:.4f} |
| **Recall** | {metrics.box.mr:.4f} |
| **mAP50** | {metrics.box.map50:.4f} |
| **mAP50-95** | {metrics.box.map:.4f} |

## Class-Specific Fine-Tuned Performance (Validation Split)
| Class ID | Class Name | Precision | Recall | mAP50 | mAP50-95 |
| :---: | :--- | :---: | :---: | :---: | :---: |
"""
        if hasattr(metrics, 'classes') and metrics.classes is not None:
            for idx, class_index in enumerate(metrics.classes):
                class_name = metrics.names[class_index]
                cp = metrics.box.p[idx]
                cr = metrics.box.r[idx]
                cmap50 = metrics.box.ap50[idx]
                cmap95 = metrics.box.ap[idx]
                md_content += f"| {class_index} | **{class_name}** | {cp:.4f} | {cr:.4f} | {cmap50:.4f} | {cmap95:.4f} |\n"

        with open(report_target_path, "w", encoding="utf-8") as f:
            f.write(md_content)
        print(f"📝 Summary performance report saved safely to: {report_target_path}")

    finally:
        # Guarantee cleanup happens regardless of exit status
        if runtime_data_yaml.exists():
            try:
                runtime_data_yaml.unlink()
            except Exception:
                pass


if __name__ == "__main__":
    train()