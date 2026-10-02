"""
split_dataset.py
=================
Creates train/val/test splits (80/10/10) from a flat images+labels directory.

Frame leakage prevention:
  - FORCE_TRAIN_PREFIXES: images whose prefix matches are assigned entirely to
    train (no splitting). Covers sequential frames and single-source datasets
    that should not bleed into val/test.
  - Remaining images are split at the group level using general prefix
    extraction so that any other sequences also stay intact across splits.

Usage:
    python split_dataset.py
    python split_dataset.py --data_dir /path/to/train --output_dir dev/data --seed 42

Before running:
    data/train/
    ├── images/   (.jpg files)
    └── labels/   (.txt files)

After running:
    dev/data/
    ├── train/images/
    ├── train/labels/
    ├── val/images/
    ├── val/labels/
    ├── test/images/
    ├── test/labels/
    └── data.yaml
"""

import os
import shutil
import random
import yaml
import argparse
from collections import defaultdict


CLASS_NAMES = {0: "bike", 1: "helmet", 2: "no-helmet", 3: "number-plate"}

# Images whose filename prefix starts with any of these strings are forced
# entirely into the train split. Reasons:
#   frame       — sequential video frames; too many to split, leakage risk
#   helmetimage — independent images from the helmetimage dataset, but kept
#                 in train for simplicity
FORCE_TRAIN_PREFIXES = ["frame", "helmetimage"]

# Classes that are rare in non-forced images (most of their annotations come
# from the forced-train groups above). Groups from remaining images that
# contain any of these classes are pushed exclusively to val/test so the
# evaluation splits have coverage of these classes.
RARE_CLASSES = {0, 3}  # bike (0), number-plate (3)


def extract_prefix(filename: str) -> str:
    """
    Extract the alphabetic prefix before the first digit in the filename.
    E.g. 'frame_162.jpg'      -> 'frame'
         'hdetect2_045.jpg'   -> 'hdetect'
         'helmetimage_01.jpg' -> 'helmetimage'
    Returns the full stem if no digit is found (treated as independent image).
    """
    name = os.path.splitext(filename)[0]
    for i, ch in enumerate(name):
        if ch.isdigit() and i > 0:
            return name[:i].rstrip("_-")
    return name


def is_force_train(filename: str) -> bool:
    prefix = extract_prefix(filename).lower()
    return any(prefix.startswith(p) for p in FORCE_TRAIN_PREFIXES)


def group_has_rare_class(files: list, labels_dir: str) -> bool:
    """Return True if any image in the group contains a RARE_CLASSES annotation."""
    for f in files:
        label_path = os.path.join(labels_dir, os.path.splitext(f)[0] + ".txt")
        if any(cls_id in RARE_CLASSES for cls_id in parse_label_file(label_path)):
            return True
    return False


def stratified_group_split(images: list, labels_dir: str,
                           train_ratio: float, val_ratio: float, seed: int):
    """
    Split images into train/val/test keeping source groups intact.

    Strategy:
      - Groups are ordered: common groups first, rare-class groups last.
        This means train fills from common groups; rare-class groups naturally
        fall into val/test slots once the train quota is met.
      - Assignment uses a deficit tracker: each group goes to whichever split
        is furthest behind its target ratio. This is robust to large groups
        and avoids the over-filling problem of simple sequential fill.
    """
    groups = defaultdict(list)
    for f in images:
        groups[extract_prefix(f)].append(f)

    random.seed(seed)

    rare_gids   = [gid for gid, files in groups.items() if group_has_rare_class(files, labels_dir)]
    common_gids = [gid for gid in groups if gid not in set(rare_gids)]

    random.shuffle(rare_gids)
    random.shuffle(common_gids)

    # Common groups first → train fills from them.
    # Rare groups last → by the time they are processed, val/test have the
    # largest deficits and rare groups are assigned there.
    ordered_gids = common_gids + rare_gids

    test_ratio = 1.0 - train_ratio - val_ratio
    targets = {"train": train_ratio, "val": val_ratio, "test": test_ratio}
    counts  = {"train": 0, "val": 0, "test": 0}
    result  = {"train": [], "val": [], "test": []}
    total   = 0

    for gid in ordered_gids:
        imgs = groups[gid]
        total += len(imgs)
        deficits = {s: targets[s] - counts[s] / total for s in targets}
        best = max(deficits, key=deficits.get)
        result[best].extend(imgs)
        counts[best] += len(imgs)

    return result["train"], result["val"], result["test"]


def parse_label_file(label_path: str) -> dict:
    class_counts = defaultdict(int)
    if not os.path.exists(label_path):
        return dict(class_counts)
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                class_counts[int(parts[0])] += 1
    return dict(class_counts)


def copy_files(file_list: list, src_images: str, src_labels: str,
               dst_images: str, dst_labels: str) -> dict:
    stats = {"created_images": 0, "created_labels": 0, "missing_labels": 0}
    for img_file in file_list:
        src_img = os.path.join(src_images, img_file)
        dst_img = os.path.join(dst_images, img_file)
        if os.path.exists(src_img):
            shutil.copy2(src_img, dst_img)
            stats["created_images"] += 1

        label_file = os.path.splitext(img_file)[0] + ".txt"
        src_label = os.path.join(src_labels, label_file)
        dst_label = os.path.join(dst_labels, label_file)
        if os.path.exists(src_label):
            shutil.copy2(src_label, dst_label)
            stats["created_labels"] += 1
        else:
            stats["missing_labels"] += 1
    return stats


def print_split_stats(name: str, files: list, labels_dir: str):
    total_counts = defaultdict(int)
    for img_file in files:
        label_path = os.path.join(labels_dir, os.path.splitext(img_file)[0] + ".txt")
        for cls_id, count in parse_label_file(label_path).items():
            total_counts[cls_id] += count
    print(f"\n  {name}: {len(files)} images")
    for cls_id in sorted(CLASS_NAMES.keys()):
        print(f"    {CLASS_NAMES[cls_id]}: {total_counts.get(cls_id, 0)} annotations")


def generate_data_yaml(output_dir: str) -> str:
    data = {
        "path": os.path.abspath(output_dir),
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "nc": len(CLASS_NAMES),
        "names": [CLASS_NAMES[i] for i in sorted(CLASS_NAMES.keys())],
    }
    yaml_path = os.path.join(output_dir, "data.yaml")
    with open(yaml_path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)
    return yaml_path


def split_dataset(
    data_dir: str,
    output_dir: str,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
):
    images_dir = os.path.join(data_dir, "images")
    labels_dir = os.path.join(data_dir, "labels")

    if not os.path.isdir(images_dir):
        raise FileNotFoundError(f"Images directory not found: {images_dir}")

    all_images = sorted([
        f for f in os.listdir(images_dir)
        if f.lower().endswith(".jpg")
    ])
    print(f"Total images found: {len(all_images)}")

    # ── Separate forced-train images ──
    forced_train = [f for f in all_images if is_force_train(f)]
    remaining    = [f for f in all_images if not is_force_train(f)]
    n = len(all_images)
    n_forced = len(forced_train)
    n_remaining = len(remaining)
    print(f"Forced into train (frame/helmetimage): {n_forced}")
    print(f"Available for splitting: {n_remaining}")

    # ── Adjust ratios so val/test hit the target fractions of the TOTAL dataset ──
    # We need:  val_count  = val_ratio  * n
    #           test_count = test_ratio * n
    # Both come entirely from `remaining`, so:
    #   adj_val_ratio  = (val_ratio  * n) / n_remaining
    #   adj_test_ratio = (test_ratio * n) / n_remaining  (implicitly handled as remainder)
    test_ratio = 1.0 - train_ratio - val_ratio
    adj_val_ratio  = (val_ratio  * n) / n_remaining
    adj_test_ratio = (test_ratio * n) / n_remaining
    adj_train_ratio = 1.0 - adj_val_ratio - adj_test_ratio

    if adj_train_ratio < 0:
        raise ValueError(
            f"Forced-train images ({n_forced}) exceed the {train_ratio:.0%} train quota "
            f"({int(train_ratio * n)} images). Reduce FORCE_TRAIN_PREFIXES or lower train_ratio."
        )

    print(f"Adjusted ratios for remaining images: "
          f"train={adj_train_ratio:.3f}  val={adj_val_ratio:.3f}  test={adj_test_ratio:.3f}")

    # ── Stratified group-aware split on remaining images ──
    split_train, val_files, test_files = stratified_group_split(
        remaining, labels_dir, adj_train_ratio, adj_val_ratio, seed
    )
    train_files = forced_train + split_train

    print(f"\nSplit results (target {train_ratio:.0%}/{val_ratio:.0%}/{test_ratio:.0%}, group-aware):")
    print(f"  train: {len(train_files):5d} ({100*len(train_files)/n:.1f}%)")
    print(f"  val:   {len(val_files):5d} ({100*len(val_files)/n:.1f}%)")
    print(f"  test:  {len(test_files):5d} ({100*len(test_files)/n:.1f}%)")

    print(f"\nAnnotation counts:")
    print_split_stats("Train", train_files, labels_dir)
    print_split_stats("Val",   val_files,   labels_dir)
    print_split_stats("Test",  test_files,  labels_dir)

    if os.path.exists(output_dir):
        print(f"\nWARNING: Output directory exists. Removing: {output_dir}")
        shutil.rmtree(output_dir)

    splits = {"train": train_files, "val": val_files, "test": test_files}
    for split_name, files in splits.items():
        img_dir = os.path.join(output_dir, split_name, "images")
        lbl_dir = os.path.join(output_dir, split_name, "labels")
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(lbl_dir, exist_ok=True)

        stats = copy_files(files, images_dir, labels_dir, img_dir, lbl_dir)
        print(f"\n{split_name}:")
        print(f"  Images copied: {stats['created_images']}")
        print(f"  Labels copied: {stats['created_labels']}")
        if stats["missing_labels"] > 0:
            print(f"  WARNING: {stats['missing_labels']} images without labels")

    yaml_path = generate_data_yaml(output_dir)
    print(f"\ndata.yaml created: {yaml_path}")
    with open(yaml_path, "r") as f:
        print(f"\n{f.read()}")
    print("Done! Dataset ready for training.")


def main():
    parser = argparse.ArgumentParser(
        description="Create group-aware 80/10/10 train/val/test splits"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=r"C:\Users\rajcr\Desktop\Study\MTech\Courses\CV\Project\Dataset\MainDataset\train",
        help="Path to directory containing images/ and labels/",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="dev/data",
        help="Path to output directory (default: dev/data)",
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.8,
        help="Fraction of splittable images for training (default: 0.8)",
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.1,
        help="Fraction of splittable images for validation (default: 0.1)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )

    args = parser.parse_args()
    split_dataset(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
