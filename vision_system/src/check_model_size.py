from pathlib import Path
from ultralytics import YOLO


# ============================================================
# CONFIGURATION
# ============================================================

# Choose which platform to check:
# "desktop" -> searches for .pt models
# "orin"    -> searches for .engine models
PLATFORM = "desktop"


# ============================================================
# PATH CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if PLATFORM == "desktop":
    MODEL_EXTENSION = "best.pt"
    SEARCH_PATH = PROJECT_ROOT

elif PLATFORM == "orin":
    MODEL_EXTENSION = "*.engine"
    SEARCH_PATH = PROJECT_ROOT / "runs" / "tensorrt_engines"

else:
    raise ValueError(
        f"Invalid PLATFORM: {PLATFORM}. "
        f"Use 'desktop' or 'orin'."
    )


# ============================================================
# FIND MODELS
# ============================================================

print(f"Platform: {PLATFORM}")
print(f"Searching under: {SEARCH_PATH}\n")

models = list(SEARCH_PATH.rglob(MODEL_EXTENSION))

if not models:
    print(f"No {MODEL_EXTENSION} models found.")
    exit()


# ============================================================
# DESKTOP: PT MODELS
# ============================================================

if PLATFORM == "desktop":

    results = []

    for model_path in models:

        print(f"Loading: {model_path}")

        # File size
        size_mb = model_path.stat().st_size / (1024 ** 2)

        # Load YOLO model
        yolo_model = YOLO(str(model_path))

        # Underlying PyTorch model
        pytorch_model = yolo_model.model

        # Count parameters
        parameter_count = sum(
            p.numel()
            for p in pytorch_model.parameters()
        )

        parameter_m = parameter_count / 1_000_000

        results.append(
            {
                "name": model_path.parent.parent.name,
                "size_mb": size_mb,
                "parameters_m": parameter_m,
                "path": model_path,
            }
        )

    # Sort by parameter count
    results.sort(key=lambda x: x["parameters_m"])

    print("\n")
    print(
        f"{'Model':<50}"
        f"{'Size (MB)':>12}"
        f"{'Params (M)':>12}"
    )

    print("-" * 76)

    for result in results:

        print(
            f"{result['name']:<50}"
            f"{result['size_mb']:>12.2f}"
            f"{result['parameters_m']:>12.2f}"
        )

    print("\nModel paths:")

    for result in results:
        print(f"  {result['path']}")

    print(f"\nTotal models found: {len(results)}")


# ============================================================
# ORIN: TENSORRT ENGINE MODELS
# ============================================================

elif PLATFORM == "orin":

    results = []

    for model_path in models:

        size_mb = model_path.stat().st_size / (1024 ** 2)

        results.append(
            {
                "name": model_path.stem,
                "size_mb": size_mb,
                "path": model_path,
            }
        )

    # Sort by engine size
    results.sort(key=lambda x: x["size_mb"])

    print("\n")
    print(
        f"{'Engine':<60}"
        f"{'Size (MB)':>12}"
    )

    print("-" * 74)

    for result in results:

        print(
            f"{result['name']:<60}"
            f"{result['size_mb']:>12.2f}"
        )

    print("\nEngine paths:")

    for result in results:
        print(f"  {result['path']}")

    print(f"\nTotal engines found: {len(results)}")