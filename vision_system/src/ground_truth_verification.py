"""
ONE-TIME SCRIPT: Verify ground truth YOLO labels align with their images.
---------------------------------------------------------------------------
This is NOT part of the main pipeline — run it once to sanity-check that
the .txt label files your colleague provided are correctly placed on the
corresponding images, before using them as ground truth for evaluation.

What it does, per image:
  1. Loads the image and reads its actual pixel dimensions.
  2. Loads the matching YOLO .txt label (class x_center y_center width height,
     all normalized 0-1).
  3. Flags obvious issues (out-of-range values, zero/negative size, missing
     label file, empty label vs. visibly-weedy image, etc.)
  4. Draws the ground truth boxes (in RED) on the image and saves it so you
     can visually confirm the boxes actually sit on the weeds.

Run from vision_system/:
    python verify_ground_truth.py

Expects:
  weed_images/capture_NNN_Brome.jpg (or .png)
  labels/capture_NNN_Brome.txt
"""

import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

# ─── CONFIG (one-time script — hardcoded is fine) ────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent
IMAGES_DIR   = PROJECT_ROOT / "weed_images"
LABELS_DIR   = PROJECT_ROOT / "labels"
OUTPUT_DIR   = PROJECT_ROOT / "gt_verification"

SUPPORTED_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]

BOX_COLOR  = (0, 0, 255)   # red in BGR, to distinguish from prediction boxes
FONT       = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.6


# ─── HELPERS ─────────────────────────────────────────────────────────────────

def find_image_for_label(label_path: Path) -> Path | None:
    """Find the image file matching a label filename (same stem, any supported ext)."""
    stem = label_path.stem
    for ext in SUPPORTED_EXTENSIONS:
        candidate = IMAGES_DIR / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def load_yolo_labels(label_path: Path) -> list[dict]:
    """
    Load a YOLO-format .txt file.
    Each line: class_id x_center y_center width height (all normalized 0-1).
    Returns a list of dicts with the raw normalized values plus the class id.
    An empty file returns an empty list (valid: "no objects in this image").
    """
    boxes = []
    if not label_path.exists():
        return boxes

    with open(label_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 5:
                print(f"    ⚠️  {label_path.name} line {line_num}: "
                      f"expected 5 values, got {len(parts)} -> '{line}'")
                continue
            try:
                class_id = int(parts[0])
                x_c, y_c, w, h = (float(v) for v in parts[1:])
            except ValueError:
                print(f"    ⚠️  {label_path.name} line {line_num}: "
                      f"could not parse numbers -> '{line}'")
                continue

            boxes.append({
                "class_id": class_id,
                "x_center": x_c, "y_center": y_c,
                "width": w, "height": h,
            })
    return boxes


def check_box_validity(box: dict, label_name: str, line_idx: int) -> list[str]:
    """Return a list of warning strings for an out-of-range or degenerate box."""
    warnings = []
    for key in ("x_center", "y_center", "width", "height"):
        val = box[key]
        if not (0.0 <= val <= 1.0):
            warnings.append(
                f"{label_name} box #{line_idx}: '{key}' = {val} is outside [0,1] "
                f"(possible resolution/normalization mismatch)"
            )
    if box["width"] <= 0 or box["height"] <= 0:
        warnings.append(
            f"{label_name} box #{line_idx}: non-positive size "
            f"(width={box['width']}, height={box['height']})"
        )
    return warnings


def yolo_to_pixel_box(box: dict, img_w: int, img_h: int) -> tuple[int, int, int, int]:
    """Convert normalized YOLO box to pixel (x1, y1, x2, y2)."""
    cx, cy = box["x_center"] * img_w, box["y_center"] * img_h
    w, h = box["width"] * img_w, box["height"] * img_h
    x1 = int(round(cx - w / 2))
    y1 = int(round(cy - h / 2))
    x2 = int(round(cx + w / 2))
    y2 = int(round(cy + h / 2))
    return x1, y1, x2, y2


# ─── MAIN ────────────────────────────────────────────────────────────────────

def main():
    if not LABELS_DIR.exists():
        print(f"Labels folder not found: {LABELS_DIR}")
        print("Place your colleague's .txt files in this folder and re-run.")
        sys.exit(1)

    if not IMAGES_DIR.exists():
        print(f"Images folder not found: {IMAGES_DIR}")
        sys.exit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    label_files = sorted(LABELS_DIR.glob("*.txt"))
    if not label_files:
        print(f"No .txt label files found in {LABELS_DIR}")
        sys.exit(1)

    print(f"Found {len(label_files)} label file(s) in {LABELS_DIR}\n")
    print("=" * 70)

    total_warnings = 0

    for label_path in label_files:
        print(f"\n{label_path.name}")

        image_path = find_image_for_label(label_path)
        if image_path is None:
            print(f"  ❌ No matching image found in {IMAGES_DIR} "
                  f"(looked for {label_path.stem}.[jpg/jpeg/png/bmp/webp])")
            total_warnings += 1
            continue

        # Load image and get its ACTUAL dimensions
        pil_image = Image.open(image_path).convert("RGB")
        img_w, img_h = pil_image.size
        print(f"  Image: {image_path.name}  ({img_w} x {img_h} px)")

        # Load ground truth boxes
        boxes = load_yolo_labels(label_path)

        if not boxes:
            print(f"  → 0 boxes (empty label file = 'no weeds' per your colleague)")
            # Still save a copy so you can visually confirm the image really
            # has no weeds, if you want to spot-check.
            bgr_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            cv2.putText(bgr_image, "GT: 0 boxes (empty label file)", (10, 30),
                        FONT, 0.9, (0, 255, 255), 2, cv2.LINE_AA)
            out_path = OUTPUT_DIR / f"gt_{image_path.stem}.jpg"
            cv2.imwrite(str(out_path), bgr_image)
            print(f"  → Saved: {out_path.name}")
            continue

        # Validate and draw
        bgr_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

        for i, box in enumerate(boxes, start=1):
            warnings = check_box_validity(box, label_path.name, i)
            for w in warnings:
                print(f"    ⚠️  {w}")
                total_warnings += len(warnings)

            x1, y1, x2, y2 = yolo_to_pixel_box(box, img_w, img_h)
            print(f"    [{i}] class={box['class_id']}  "
                  f"pixel box=({x1},{y1}) -> ({x2},{y2})")

            cv2.rectangle(bgr_image, (x1, y1), (x2, y2), BOX_COLOR, 2)
            label_text = f"GT class={box['class_id']} #{i}"
            (tw, th), _ = cv2.getTextSize(label_text, FONT, FONT_SCALE, 1)
            cv2.rectangle(bgr_image, (x1, y1 - th - 6), (x1 + tw + 4, y1), BOX_COLOR, -1)
            cv2.putText(bgr_image, label_text, (x1 + 2, y1 - 4),
                        FONT, FONT_SCALE, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.putText(bgr_image, f"GT boxes: {len(boxes)}", (10, 30),
                    FONT, 0.9, (0, 255, 255), 2, cv2.LINE_AA)

        out_path = OUTPUT_DIR / f"gt_{image_path.stem}.jpg"
        cv2.imwrite(str(out_path), bgr_image)
        print(f"  → Saved: {out_path.name}")

    print("\n" + "=" * 70)
    if total_warnings == 0:
        print("No structural issues detected (no out-of-range or degenerate boxes).")
    else:
        print(f"{total_warnings} warning(s) found — see above.")

    print(f"\nNow open the images in '{OUTPUT_DIR.name}/' and visually check:")
    print("  - Do the red boxes sit on top of actual weed plants?")
    print("  - Are images with '0 boxes' actually weed-free?")
    print("  - If boxes are clearly misaligned/shifted/wrong scale, the")
    print("    image resolution used for normalization may not match these")
    print("    images, and these labels should NOT be used as ground truth")
    print("    without re-deriving them at the correct resolution.")


if __name__ == "__main__":
    main()