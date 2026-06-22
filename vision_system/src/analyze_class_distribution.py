"""
ONE-TIME SCRIPT: Analyze class distribution across the training split.
---------------------------------------------------------------------------
Reads YOLO-format label .txt files under labels/train and evaluates:
1. Training Image Distribution (Unique images containing classes vs empty backgrounds)
2. Training Instance Distribution (Total individual objects of the class)

Displays both metrics side-by-side using clean ring/donut charts.

Run from vision_system/:
    python src/analyze_class_distribution.py
"""

import sys
from pathlib import Path
from collections import Counter

import yaml
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_YAML_PATH = PROJECT_ROOT / "config" / "data.yaml"

OUTPUT_CHART = PROJECT_ROOT / "class_distribution.png"


def load_class_names() -> tuple[dict, dict]:
    with open(DATA_YAML_PATH, "r", encoding="utf-8") as f:
        data_cfg = yaml.safe_load(f)
    return data_cfg.get("names", {}), data_cfg


def resolve_train_label_dir(data_cfg: dict) -> Path:
    dataset_root = (DATA_YAML_PATH.parent / data_cfg["path"]).resolve()
    split_rel_path = data_cfg.get("train", "images/train")
    img_dir = (dataset_root / split_rel_path).resolve()
    
    label_dir = Path(str(img_dir).replace("images", "labels", 1))
    return label_dir


def analyze_training_split(label_dir: Path) -> tuple[Counter, Counter, int]:
    """
    Computes unique image counts per class (including a separate count for empty background images)
    and total instance counts per class strictly inside the training partition.
    """
    instance_counts = Counter()
    image_counts = Counter()
    empty_background_count = 0

    if not label_dir.exists():
        return image_counts, instance_counts, empty_background_count

    for txt_file in label_dir.glob("*.txt"):
        try:
            with open(txt_file, "r", encoding="utf-8") as f:
                lines = [line.strip() for line in f if line.strip()]
                
            if not lines:
                # File is empty -> Explicit background sample
                empty_background_count += 1
                continue

            classes_in_this_image = set()
            for line in lines:
                parts = line.split()
                if parts:
                    class_id = int(parts[0])
                    instance_counts[class_id] += 1
                    classes_in_this_image.add(class_id)
            
            # Record unique image presence per weed class
            for class_id in classes_in_this_image:
                image_counts[class_id] += 1
                
        except Exception as e:
            print(f"Error reading {txt_file.name}: {e}")
            
    return image_counts, instance_counts, empty_background_count


def save_dual_distribution_donuts(label_dir: Path, image_counts: Counter, instance_counts: Counter, 
                                  empty_background_count: int, class_names: dict, output_path: Path):
    """Generates a side-by-side subplot figure displaying both ring charts."""
    if not image_counts and not instance_counts and empty_background_count == 0:
        print("❌ Error: No training data found. Cannot generate charts.")
        return

    # Create a unified high-resolution figure with two side-by-side subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 9), facecolor="white")
    
    # Generate an adaptive color map palette based on the total number of unique classes found
    unique_classes = sorted(list(set(image_counts.keys()).union(set(instance_counts.keys()))))
    color_cycle = plt.cm.tab20(np.linspace(0, 1, len(unique_classes) + 1)) # +1 to save a color slot for background
    
    class_color_map = {cid: color_cycle[i] for i, cid in enumerate(unique_classes)}
    background_color = '#95a5a6'  # Neutral elegant grey for empty background samples

    # Helper to generate labeling text format string: displays "Count\n(Pct%)" or stays empty if slice is tiny
    def make_autopct_with_values(total_sum):
        def inner_autopct(pct):
            if pct < 1.5:  # Hides labels on small slices to avoid overlapping text
                return ''
            val = int(round(pct * total_sum / 100.0))
            return f"{val:,}\n({pct:.1f}%)"
        return inner_autopct

    # Calculate actual total text files found on disk to verify data matches perfectly
    actual_total_files = len(list(label_dir.glob("*.txt")))

    # ─── LEFT SUBPLOT: IMAGE DISTRIBUTION (Includes Background) ─────────────
    img_sorted = image_counts.most_common()
    img_cids = [item[0] for item in img_sorted]
    img_sizes = [item[1] for item in img_sorted]
    img_labels = [class_names.get(cid, f"Class {cid}") for cid in img_cids]
    img_colors = [class_color_map[cid] for cid in img_cids]

    # Append the explicit Background class if background files are present
    if empty_background_count > 0:
        img_labels.append("BACKGROUND (Empty)")
        img_sizes.append(empty_background_count)
        img_colors.append(background_color)

    total_imgs_sum = sum(img_sizes)

    wedges1, texts1, autotexts1 = ax1.pie(
        img_sizes,
        labels=img_labels,
        autopct=make_autopct_with_values(total_imgs_sum),
        startangle=140,
        colors=img_colors,
        pctdistance=0.72,
        labeldistance=1.08,
        wedgeprops=dict(width=0.35, edgecolor='w', linewidth=1.5)
    )
    ax1.set_title(f"Training Dataset Image Distribution\n(Total Disk Files: {actual_total_files:,} | Sum of Slices: {total_imgs_sum:,})", 
                  fontsize=13, weight='bold', pad=15, color='#2c3e50')
    ax1.axis('equal')

    # ─── RIGHT SUBPLOT: INSTANCE DISTRIBUTION (Objects Only) ────────────────
    inst_sorted = instance_counts.most_common()
    inst_cids = [item[0] for item in inst_sorted]
    inst_sizes = [item[1] for item in inst_sorted]
    inst_labels = [class_names.get(cid, f"Class {cid}") for cid in inst_cids]
    inst_colors = [class_color_map[cid] for cid in inst_cids]
    total_inst_sum = sum(inst_sizes)

    wedges2, texts2, autotexts2 = ax2.pie(
        inst_sizes,
        labels=inst_labels,
        autopct=make_autopct_with_values(total_inst_sum),
        startangle=140,
        colors=inst_colors,
        pctdistance=0.72,
        labeldistance=1.08,
        wedgeprops=dict(width=0.35, edgecolor='w', linewidth=1.5)
    )
    ax2.set_title(f"Training Dataset Instance Distribution\n(Total Labeled Object Boxes: {total_inst_sum:,})", 
                  fontsize=13, weight='bold', pad=15, color='#2c3e50')
    ax2.axis('equal')

    # ─── STYLING & AESTHETICS ───────────────────────────────────────────────
    for t in texts1 + texts2:
        t.set_color('#34495e')
        t.set_fontsize(9.5)
    for at in autotexts1 + autotexts2:
        at.set_color('white')
        at.set_weight('bold')
        at.set_fontsize(8.5)

    plt.suptitle("AI-Sprayer Pipeline: Labeled Training Split Overview Matrix", 
                 fontsize=16, weight='bold', y=0.98, color='#2c3e50')
    
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output_path, dpi=150)
    print(f"\n📊 Dual-donut distribution overview saved successfully to: {output_path}")


def main():
    if not DATA_YAML_PATH.exists():
        print(f"data.yaml not found: {DATA_YAML_PATH}")
        sys.exit(1)

    class_names, data_cfg = load_class_names()
    label_dir = resolve_train_label_dir(data_cfg)

    print("=" * 70)
    print(f"Analyzing Training Split Distribution Profiles from: {label_dir.name}")
    print("=" * 70)

    image_counts, instance_counts, empty_background_count = analyze_training_split(label_dir)

    actual_total_files = len(list(label_dir.glob('*.txt')))
    print(f"  -> Total Label Files on Disk: {actual_total_files:,}")
    print(f"  -> Explicit Background (Empty Files): {empty_background_count:,}")
    print(f"  -> Total Labeled Bounding Boxes: {sum(instance_counts.values()):,}")

    save_dual_distribution_donuts(label_dir, image_counts, instance_counts, empty_background_count, class_names, OUTPUT_CHART)


if __name__ == "__main__":
    main()