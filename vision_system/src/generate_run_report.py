"""
Retroactive Run Report Generator
---------------------------------
Loads an existing trained model checkpoint, executes a quick evaluation pass,
and generates the corresponding 'summaried_performance.md' report file layout instantly.
"""

import sys
from pathlib import Path
import yaml
import tempfile

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Loads train config to resolve your dataset and validation threshold parameters safely
from src.train_yolo import load_train_config, build_runtime_data_yaml

def generate_retroactive_report(run_folder_path: str):
    run_dir = Path(run_folder_path).resolve()
    best_weights = run_dir / "weights" / "best.pt"
    
    if not best_weights.exists():
        raise FileNotFoundError(f"Could not locate model weights anchor at: {best_weights}")
        
    cfg = load_train_config()
    runtime_data_yaml = build_runtime_data_yaml(cfg["DATA_YAML"])

    print(f"Loading trained weights from: {best_weights}")
    from ultralytics import YOLO
    model = YOLO(str(best_weights))


    print("Running evaluation baseline pass over validation split...")
    metrics = model.val(
        data=str(runtime_data_yaml),
        iou=cfg["VAL_IOU"],
        conf=cfg["VAL_CONF"],
        project=str(run_dir.parent),
        name=f"{run_dir.name}/val",
        exist_ok=True,
        verbose=True, 
        plots=True
    )

    # Build report path
    report_target_path = run_dir / "summaried_performance.md"
    
    md_content = f"""# YOLOv8 Training Run Performance Summary
**Run Target Directory:** `{run_dir.name}`  
**Model Weight Checkpoint:** `{best_weights.name}` (Generated Retroactively)

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
        
    print(f"✅ Success! Retroactive performance summary report written to: {report_target_path}")

    try:
        runtime_data_yaml.unlink()
    except Exception:
        pass


if __name__ == "__main__":
    # Provide the path to your historical training run directory here:
    TARGET_RUN = "D:/MyProjects/AI-Sprayer/vision_system/runs/train/yolov8n_ep20_b4_lr0.0002"
    generate_retroactive_report(TARGET_RUN)