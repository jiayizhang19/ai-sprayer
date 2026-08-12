from pathlib import Path
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[2]

print(f"Searching under: {PROJECT_ROOT}\n")

models = list(PROJECT_ROOT.rglob("best.pt"))

if not models:
    print("No best.pt models found.")
    exit()

results = []

for model_path in models:

    print(f"Loading: {model_path}")

    # Load model using Ultralytics
    yolo_model = YOLO(str(model_path))

    # Underlying PyTorch model
    pytorch_model = yolo_model.model

    # Count parameters
    parameter_count = sum(
        p.numel()
        for p in pytorch_model.parameters()
    )

    parameter_m = parameter_count / 1_000_000

    # Model file size
    size_mb = model_path.stat().st_size / (1024 ** 2)

    results.append(
        {
            "name": model_path.parent.parent.name,
            "size_mb": size_mb,
            "parameters": parameter_count,
            "parameters_m": parameter_m,
            "path": model_path,
        }
    )

# Sort by parameter count
results.sort(key=lambda x: x["parameters"])

print("\n")
print(f"{'Model':<45} {'Size (MB)':>12} {'Params (M)':>12}")
print("-" * 72)

for result in results:

    print(
        f"{result['name']:<45}"
        f"{result['size_mb']:>12.2f}"
        f"{result['parameters_m']:>12.2f}"
    )

print("\nModel paths:")

for result in results:
    print(f"  {result['path']}")

print(f"\nTotal models found: {len(results)}")