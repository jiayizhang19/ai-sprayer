"""
ONE-TIME SCRIPT: Analyze class distribution across train/val/test splits.
---------------------------------------------------------------------------
Reads YOLO-format label .txt files (class_id x_center y_center width height)
under labels/train, labels/val, labels/test and reports how many instances
of each class appear in each split. Useful for spotting class imbalance
before training (e.g. one weed species dominating the dataset).

Not part of the live detection/training pipeline — run standalone whenever
you want to re-check the dataset.

Run from vision_system/:
    python src/analyze_class_distribution.py

Expects (Ultralytics standard layout):
    labels/train/*.txt
    labels/val/*.txt
    labels/test/*.txt
Class names are read from config/data.yaml so output uses class codes
(e.g. BROST) rather than bare IDs.
"""

import sys
from pathlib import Path
from collections import Counter

import yaml
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_YAML_PATH = PROJECT_ROOT / "config" / "data.yaml"

# Splits to analyze and where their label folders live, relative to the
# dataset root declared in data.yaml ("path:" + "train:"/"val:"/"test:",
# with "images" swapped for "labels").
SPLITS = ["train", "val", "test"]

OUTPUT_CHART = PROJECT_ROOT / "class_distribution.png"


def load_class_names() -> dict:
    with open(DATA_YAML_PATH, "r", encoding="utf-8") as f:
        data_cfg = yaml.safe_load(f)
    return {int(k): v for k, v in data_cfg["names"].items()}, data_cfg


def resolve_label_dir(data_cfg: dict, split: str) -> Path:
    """Convert the images/<split> path from data.yaml into labels/<split>."""
    dataset_root = (DATA_YAML_PATH.parent / data_cfg["path"]).resolve()
    images_rel = data_cfg.get(split)
    if not images_rel:
        return None
    images_dir = (dataset_root / images_rel).resolve()
    labels_dir = Path(str(images_dir).replace("images", "labels", 1))
    return labels_dir


def count_classes_in_split(label_dir: Path) -> Counter:
    """Count class instances (one count per bounding box line) in a split."""
    counts = Counter()
    if not label_dir or not label_dir.exists():
        return counts

    for txt_file in label_dir.glob("*.txt"):
        with open(txt_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) != 5:
                    continue
                try:
                    class_id = int(parts[0])
                except ValueError:
                    continue
                counts[class_id] += 1

    return counts


def print_table(counts_by_split: dict, class_names: dict):
    all_class_ids = sorted(set().union(*[c.keys() for c in counts_by_split.values()]))

    if not all_class_ids:
        print("No labeled instances found in any split.")
        return

    name_width = max(len(class_names.get(cid, f"id_{cid}")) for cid in all_class_ids)
    name_width = max(name_width, len("Class"))

    header = f"{'Class':<{name_width}}  " + "  ".join(f"{s:>8}" for s in SPLITS) + f"  {'Total':>8}"
    print(header)
    print("-" * len(header))

    split_totals = {s: 0 for s in SPLITS}
    for cid in all_class_ids:
        name = class_names.get(cid, f"id_{cid}")
        row_counts = [counts_by_split[s].get(cid, 0) for s in SPLITS]
        row_total = sum(row_counts)
        for s, c in zip(SPLITS, row_counts):
            split_totals[s] += c
        row = f"{name:<{name_width}}  " + "  ".join(f"{c:>8}" for c in row_counts) + f"  {row_total:>8}"
        print(row)

    print("-" * len(header))
    totals_row = f"{'TOTAL':<{name_width}}  " + "  ".join(f"{split_totals[s]:>8}" for s in SPLITS) \
                 + f"  {sum(split_totals.values()):>8}"
    print(totals_row)

    print(f"\nImages with zero labeled instances (empty .txt files) are not "
          f"counted above — they represent 'no weeds present' images.")


def plot_chart(counts_by_split: dict, class_names: dict, output_path: Path):
    all_class_ids = sorted(set().union(*[c.keys() for c in counts_by_split.values()]))
    if not all_class_ids:
        print("Skipping chart: no labeled instances to plot.")
        return

    labels = [class_names.get(cid, f"id_{cid}") for cid in all_class_ids]
    x = np.arange(len(labels))
    bar_width = 0.25

    fig, ax = plt.subplots(figsize=(max(10, len(labels) * 0.5), 6))

    for i, split in enumerate(SPLITS):
        values = [counts_by_split[split].get(cid, 0) for cid in all_class_ids]
        offset = (i - 1) * bar_width
        ax.bar(x + offset, values, bar_width, label=split)

    ax.set_xlabel("Class")
    ax.set_ylabel("Instance count")
    ax.set_title("Class Distribution by Split")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    print(f"\nChart saved to: {output_path}")


def main():
    if not DATA_YAML_PATH.exists():
        print(f"data.yaml not found: {DATA_YAML_PATH}")
        sys.exit(1)

    class_names, data_cfg = load_class_names()

    counts_by_split = {}
    for split in SPLITS:
        label_dir = resolve_label_dir(data_cfg, split)
        counts = count_classes_in_split(label_dir)
        counts_by_split[split] = counts
        found_msg = f"{label_dir}" if label_dir and label_dir.exists() else f"{label_dir} (not found)"
        print(f"{split}: {sum(counts.values())} instances across "
              f"{len(list(label_dir.glob('*.txt'))) if label_dir and label_dir.exists() else 0} label files  [{found_msg}]")

    print()
    print_table(counts_by_split, class_names)
    plot_chart(counts_by_split, class_names, OUTPUT_CHART)


if __name__ == "__main__":
    main()
