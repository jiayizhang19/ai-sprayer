"""
Check image resolutions in the test set and group them by size.
Just run: python check_image_resolutions.py
"""

from pathlib import Path
from collections import defaultdict
from PIL import Image

# ============================================================
# CONFIG – change this path if needed
# ============================================================
IMAGES_DIR = Path("/home/aura/jiayi/ai-sprayer/vision_system/images/test")   # ← edit this if necessary
# ============================================================


def main():
    images_dir = IMAGES_DIR

    if not images_dir.exists():
        print(f"❌ Folder not found: {images_dir}")
        return

    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    resolution_groups = defaultdict(list)
    total = 0

    print(f"Scanning images in: {images_dir}\n")

    for img_path in sorted(images_dir.iterdir()):
        if img_path.suffix.lower() not in extensions:
            continue

        try:
            with Image.open(img_path) as img:
                w, h = img.size
                resolution_groups[(w, h)].append(img_path.name)
                total += 1
        except Exception as e:
            print(f"⚠️  Could not open {img_path.name}: {e}")

    print(f"Total images scanned: {total}\n")
    print("=" * 60)
    print(f"{'Resolution':<20} {'Count':<10} {'Percentage'}")
    print("=" * 60)

    sorted_groups = sorted(resolution_groups.items(), key=lambda x: len(x[1]), reverse=True)

    for (w, h), files in sorted_groups:
        count = len(files)
        pct = count / total * 100
        print(f"{w} x {h:<12} {count:<10} {pct:.1f}%")

    print("=" * 60)

    print("\nExample filenames per resolution group:\n")
    for (w, h), files in sorted_groups[:5]:
        print(f"→ {w}x{h} ({len(files)} images):")
        for name in files[:3]:
            print(f"   - {name}")
        if len(files) > 3:
            print(f"   ... and {len(files)-3} more")
        print()


if __name__ == "__main__":
    main()