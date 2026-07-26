"""
Export YOLO models to TensorRT (FP16 / INT8) or ONNX (dry-run)
--------------------------------------------------------------
- On Windows  → use --dry-run (exports ONNX for testing)
- On Orin     → run without --dry-run (real TensorRT engines)

Folder structure:
  runs/
  ├── tensorrt_engines/     ← .engine files (Orin)
  └── onnx/                 ← .onnx files (Windows dry-run)

Commands on Windows (dry-run):
  python export_tensorrt.py --dry-run
Commands on Orin (real TensorRT export):
  python export_tensorrt.py --precision fp16
"""

from pathlib import Path
from ultralytics import YOLO
import argparse

# ============================================================
# CONFIG
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODELS_TO_EXPORT = [
    {
        "name": "yolo26s_ep150_b8_lr0.001",
        "pt_path": PROJECT_ROOT / "runs/train/yolo26s_ep150_b8_lr0.001/weights/best.pt",
    },
    {
        "name": "yolo26m_ep150_b8_lr0.001",
        "pt_path": PROJECT_ROOT / "runs/train/yolo26m_ep150_b8_lr0.001/weights/best.pt",
    },
    {
        "name": "yolov8m_ep150_b8_lr0.001",
        "pt_path": PROJECT_ROOT / "runs/train/yolov8m_ep150_b8_lr0.001/weights/best.pt",
    },
]

TENSORRT_DIR = PROJECT_ROOT / "runs" / "tensorrt_engines"
ONNX_DIR     = PROJECT_ROOT / "runs" / "onnx"
IMGSZ        = 800
DEVICE       = 0


def export_model(pt_path: Path, name: str, precision: str = "fp16", dry_run: bool = False):
    if not pt_path.exists():
        print(f"❌ Model not found: {pt_path}")
        return None

    mode = "ONNX (dry-run)" if dry_run else f"TensorRT {precision.upper()}"
    print(f"\n{'='*70}")
    print(f"Exporting: {name}  →  {mode}")
    print(f"Source: {pt_path}")
    print(f"{'='*70}")

    model = YOLO(str(pt_path))

    if dry_run:
        # ---------- Safe path for Windows ----------
        export_args = {
            "format": "onnx",
            "imgsz": IMGSZ,
            "simplify": True,
            "opset": 18,
            "verbose": True,
        }
        out_path = model.export(**export_args)
        out_path = Path(out_path)

        ONNX_DIR.mkdir(parents=True, exist_ok=True)
        final_path = ONNX_DIR / f"{name}.onnx"
        if out_path != final_path:
            out_path.replace(final_path)

        print(f"✅ ONNX exported → {final_path}")
        return final_path

    else:
        # ---------- Real TensorRT export (Orin only) ----------
        export_args = {
            "format": "engine",
            "imgsz": IMGSZ,
            "device": DEVICE,
            "half": precision == "fp16",
            "int8": precision == "int8",
            "workspace": 4,
            "verbose": True,
        }

        if precision == "int8":
            data_yaml = PROJECT_ROOT / "config" / "data.yaml"
            if data_yaml.exists():
                export_args["data"] = str(data_yaml)

        engine_path = model.export(**export_args)
        engine_path = Path(engine_path)

        TENSORRT_DIR.mkdir(parents=True, exist_ok=True)
        final_path = TENSORRT_DIR / f"{name}_{precision}.engine"
        if engine_path != final_path:
            engine_path.replace(final_path)

        print(f"✅ TensorRT engine exported → {final_path}")
        return final_path


def main():
    parser = argparse.ArgumentParser(description="Export YOLO models for Jetson Orin")
    parser.add_argument(
        "--precision",
        choices=["fp16", "int8", "both"],
        default="fp16",
        help="Precision for TensorRT (ignored in dry-run)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Safe mode: only export ONNX (works on Windows)",
    )
    args = parser.parse_args()

    print("=" * 70)
    if args.dry_run:
        print("DRY-RUN MODE (Windows-safe) → Exporting ONNX only")
        print(f"Output directory: {ONNX_DIR}")
    else:
        print("REAL TensorRT EXPORT (run this on Jetson Orin)")
        print(f"Precision: {args.precision}")
        print(f"Output directory: {TENSORRT_DIR}")
    print("=" * 70)

    results = []

    for m in MODELS_TO_EXPORT:
        if args.dry_run:
            path = export_model(m["pt_path"], m["name"], dry_run=True)
            if path:
                results.append(path)
        else:
            if args.precision in ("fp16", "both"):
                path = export_model(m["pt_path"], m["name"], precision="fp16", dry_run=False)
                if path:
                    results.append(path)
            if args.precision in ("int8", "both"):
                path = export_model(m["pt_path"], m["name"], precision="int8", dry_run=False)
                if path:
                    results.append(path)

    print("\n" + "=" * 70)
    print("Export finished. Generated files:")
    for p in results:
        print(f"  • {p}")
    print("=" * 70)

    if args.dry_run:
        print("\n✅ Dry-run completed successfully.")
        print("You can now transfer the script + .pt files to the Orin and run without --dry-run.")
    else:
        print("\nNext step on Orin:")
        print("1. Point config.yaml → evaluation.eval_yolo_model to the .engine file")
        print("2. Re-run detection_evaluation.py")


if __name__ == "__main__":
    main()