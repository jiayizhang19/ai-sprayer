"""
AISprayer Class Distribution Analyzer - FINAL FIXED VERSION
"""

import sys
from pathlib import Path
from collections import Counter
import csv

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
AISPRAYER_BBOX_DIR = Path(r"D:\MyProjects\cropandweed-dataset\data\bboxes\AISprayer")

OUTPUT_CHART = PROJECT_ROOT / "class_distribution_cropandweed.png"   # Updated filename as requested

CLASS_NAMES = {
    0: "Brome",
    1: "Chickweed",
    2: "Nettle",
    3: "OtherWeed",
}


def resolve_aisprayer_bbox_dir() -> Path:
    if AISPRAYER_BBOX_DIR.exists():
        csv_count = len(list(AISPRAYER_BBOX_DIR.glob("*.csv")))
        print(f"✅ Found AISprayer: {AISPRAYER_BBOX_DIR} ({csv_count} files)")
        return AISPRAYER_BBOX_DIR
    print("❌ Path not found!")
    sys.exit(1)


def analyze_aisprayer_split(label_dir: Path) -> tuple[Counter, Counter, int]:
    instance_counts = Counter()
    image_counts = Counter()
    empty_background_count = 0

    csv_files = list(label_dir.glob("*.csv"))
    print(f"📊 Processing {len(csv_files)} CSV files...")

    for csv_file in csv_files:
        try:
            classes_in_this_image = set()

            with open(csv_file, "r", newline='', encoding="utf-8") as f:
                reader = csv.reader(f)  # No header
                
                for row in reader:
                    if len(row) < 5:
                        continue
                    try:
                        label_id = int(row[4])   # label_id is 5th column (0-based index 4)
                        if label_id in CLASS_NAMES:
                            instance_counts[label_id] += 1
                            classes_in_this_image.add(label_id)
                    except (ValueError, IndexError):
                        continue

            if not classes_in_this_image:
                empty_background_count += 1
            else:
                for class_id in classes_in_this_image:
                    image_counts[class_id] += 1

        except Exception as e:
            print(f"Error reading {csv_file.name}: {e}")

    return image_counts, instance_counts, empty_background_count


def save_dual_distribution_donuts(label_dir: Path, image_counts: Counter, instance_counts: Counter, 
                                  empty_background_count: int, output_path: Path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 9), facecolor="white")
    
    unique_classes = sorted(list(set(image_counts.keys()).union(set(instance_counts.keys()))))
    color_cycle = plt.cm.tab20(np.linspace(0, 1, len(unique_classes) + 1))
    class_color_map = {cid: color_cycle[i] for i, cid in enumerate(unique_classes)}
    background_color = '#95a5a6'

    def make_autopct(total):
        def inner(pct):
            if pct < 1.5: return ''
            val = int(round(pct * total / 100))
            return f"{val:,}\n({pct:.1f}%)"
        return inner

    actual_total = len(list(label_dir.glob("*.csv")))

    # Left: Image Distribution
    img_sorted = image_counts.most_common()
    img_sizes = [c for _,c in img_sorted]
    img_labels = [CLASS_NAMES.get(cid, str(cid)) for cid,_ in img_sorted]
    img_colors = [class_color_map[cid] for cid,_ in img_sorted]

    if empty_background_count > 0:
        img_labels.append("BACKGROUND (Empty)")
        img_sizes.append(empty_background_count)
        img_colors.append(background_color)

    ax1.pie(img_sizes, labels=img_labels, autopct=make_autopct(sum(img_sizes)),
            startangle=140, colors=img_colors, pctdistance=0.72, labeldistance=1.05,
            wedgeprops=dict(width=0.35, edgecolor='w'))
    ax1.set_title(f"AISprayer (CropAndWeed) - Image Distribution\n(Total Files: {actual_total:,})", 
                  fontsize=12, weight='bold')

    # Right: Instance Distribution
    inst_sorted = instance_counts.most_common()
    inst_sizes = [c for _,c in inst_sorted]
    inst_labels = [CLASS_NAMES.get(cid, str(cid)) for cid,_ in inst_sorted]
    inst_colors = [class_color_map[cid] for cid,_ in inst_sorted]

    ax2.pie(inst_sizes, labels=inst_labels, autopct=make_autopct(sum(inst_sizes)),
            startangle=140, colors=inst_colors, pctdistance=0.72, labeldistance=1.05,
            wedgeprops=dict(width=0.35, edgecolor='w'))
    ax2.set_title(f"AISprayer (CropAndWeed) - Instance Distribution\n(Total Boxes: {sum(inst_sizes):,})", 
                  fontsize=12, weight='bold')

    plt.suptitle("AISprayer (from CropAndWeed) - Class Distribution", fontsize=16, weight='bold', y=0.95)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ Chart saved as: {output_path}")


def main():
    label_dir = resolve_aisprayer_bbox_dir()
    image_counts, instance_counts, empty = analyze_aisprayer_split(label_dir)

    print("\n" + "="*80)
    print("AISprayer (CropAndWeed) Analysis Summary")
    print("="*80)
    print(f"Total Files       : {len(list(label_dir.glob('*.csv'))):,}")
    print(f"Empty Files       : {empty:,}")
    print(f"Files with boxes  : {len(image_counts):,}")
    print(f"Total Boxes       : {sum(instance_counts.values()):,}\n")

    for cid in sorted(CLASS_NAMES.keys()):
        print(f"  {CLASS_NAMES[cid]:12} : {image_counts[cid]:6,} images | {instance_counts[cid]:7,} boxes")

    save_dual_distribution_donuts(label_dir, image_counts, instance_counts, empty, OUTPUT_CHART)


if __name__ == "__main__":
    main()