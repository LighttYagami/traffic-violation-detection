"""
verify_dataset.py
==================
Sanity checks on the processed dataset before training.
Catches corrupt images, bad labels, class imbalance, and missing files.

Usage:
    python verify_dataset.py --data_dir /path/to/data/processed

Checks performed:
    1. Every image can be loaded by OpenCV
    2. Every label file has valid YOLO format (class_id cx cy w h, normalized)
    3. Every image has a matching label file (and vice versa)
    4. Class IDs are within expected range
    5. Bounding box coordinates are normalized (0-1)
    6. No empty label files (warns, doesn't fail)
    7. Class distribution per split
    8. Bbox size distribution (flags tiny boxes < 10x10 pixels)
    9. Train/val/test split ratios
"""

import os
import cv2
import argparse
import numpy as np
from collections import defaultdict, Counter
from pathlib import Path


# Must match your data.yaml
CLASS_NAMES = {0: "bike", 1: "helmet", 2: "no-helmet", 3: "number-plate"}
NUM_CLASSES = len(CLASS_NAMES)
ASSUMED_IMG_SIZE = 1280  # for estimating pixel-level bbox sizes


def check_image(image_path: str) -> tuple:
    """
    Try loading an image. Returns (ok, width, height) or (False, 0, 0).
    """
    try:
        img = cv2.imread(image_path)
        if img is None:
            return False, 0, 0
        h, w = img.shape[:2]
        return True, w, h
    except Exception:
        return False, 0, 0


def check_label(label_path: str) -> dict:
    """
    Validate a YOLO label file.
    Returns dict with: valid, num_annotations, class_ids, errors, warnings, bboxes
    """
    result = {
        "valid": True,
        "num_annotations": 0,
        "class_ids": [],
        "errors": [],
        "warnings": [],
        "bboxes": [],  # list of (class_id, cx, cy, w, h)
    }

    if not os.path.exists(label_path):
        result["valid"] = False
        result["errors"].append("File not found")
        return result

    with open(label_path, "r") as f:
        lines = f.readlines()

    if len(lines) == 0:
        result["warnings"].append("Empty label file")
        return result

    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue

        parts = line.split()

        # Check format: class_id cx cy w h
        if len(parts) < 5:
            result["valid"] = False
            result["errors"].append(f"Line {i+1}: expected 5 values, got {len(parts)}")
            continue

        try:
            class_id = int(parts[0])
            cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
        except ValueError:
            result["valid"] = False
            result["errors"].append(f"Line {i+1}: non-numeric values")
            continue

        # Check class ID range
        if class_id < 0 or class_id >= NUM_CLASSES:
            result["valid"] = False
            result["errors"].append(
                f"Line {i+1}: class_id {class_id} out of range [0, {NUM_CLASSES-1}]"
            )

        # Check normalized coordinates
        for name, val in [("cx", cx), ("cy", cy), ("w", w), ("h", h)]:
            if val < 0 or val > 1:
                result["valid"] = False
                result["errors"].append(
                    f"Line {i+1}: {name}={val:.4f} not in [0, 1]"
                )

        # Check for tiny bboxes
        pixel_w = w * ASSUMED_IMG_SIZE
        pixel_h = h * ASSUMED_IMG_SIZE
        if pixel_w < 10 or pixel_h < 10:
            result["warnings"].append(
                f"Line {i+1}: tiny bbox ({pixel_w:.0f}x{pixel_h:.0f}px at {ASSUMED_IMG_SIZE}px)"
            )

        result["num_annotations"] += 1
        result["class_ids"].append(class_id)
        result["bboxes"].append((class_id, cx, cy, w, h))

    return result


def verify_split(split_name: str, split_dir: str) -> dict:
    """
    Verify all images and labels in a split.
    Returns summary dict.
    """
    images_dir = os.path.join(split_dir, "images")
    labels_dir = os.path.join(split_dir, "labels")

    if not os.path.isdir(images_dir):
        print(f"  WARNING: {images_dir} does not exist")
        return {}

    # Get file lists
    image_files = set(
        os.path.splitext(f)[0]
        for f in os.listdir(images_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    )
    label_files = set(
        os.path.splitext(f)[0]
        for f in os.listdir(labels_dir)
        if f.lower().endswith(".txt")
    ) if os.path.isdir(labels_dir) else set()

    # Check for mismatches
    images_without_labels = image_files - label_files
    labels_without_images = label_files - image_files

    summary = {
        "total_images": len(image_files),
        "total_labels": len(label_files),
        "images_without_labels": len(images_without_labels),
        "labels_without_images": len(labels_without_images),
        "corrupt_images": 0,
        "invalid_labels": 0,
        "empty_labels": 0,
        "total_annotations": 0,
        "class_counts": defaultdict(int),
        "tiny_bboxes": 0,
        "image_sizes": [],
        "errors": [],
        "warnings": [],
    }

    print(f"\n{'='*60}")
    print(f"  Verifying: {split_name} ({len(image_files)} images)")
    print(f"{'='*60}")

    if images_without_labels:
        summary["warnings"].append(
            f"{len(images_without_labels)} images without labels"
        )
        if len(images_without_labels) <= 5:
            for f in images_without_labels:
                print(f"    Missing label: {f}")

    if labels_without_images:
        summary["warnings"].append(
            f"{len(labels_without_images)} labels without images"
        )

    # Check each image
    for stem in sorted(image_files):
        # Find actual image file
        img_path = None
        for ext in [".jpg", ".jpeg", ".png"]:
            candidate = os.path.join(images_dir, stem + ext)
            if os.path.exists(candidate):
                img_path = candidate
                break

        if img_path is None:
            continue

        # Verify image loads
        ok, w, h = check_image(img_path)
        if not ok:
            summary["corrupt_images"] += 1
            summary["errors"].append(f"Corrupt image: {stem}")
            continue
        summary["image_sizes"].append((w, h))

        # Verify label
        label_path = os.path.join(labels_dir, stem + ".txt")
        label_result = check_label(label_path)

        if not label_result["valid"]:
            summary["invalid_labels"] += 1
            for err in label_result["errors"]:
                summary["errors"].append(f"{stem}: {err}")

        if label_result["warnings"]:
            for warn in label_result["warnings"]:
                if "Empty" in warn:
                    summary["empty_labels"] += 1
                if "tiny" in warn:
                    summary["tiny_bboxes"] += 1

        summary["total_annotations"] += label_result["num_annotations"]
        for cls_id in label_result["class_ids"]:
            summary["class_counts"][cls_id] += 1

    # Print results
    print(f"  Images:        {summary['total_images']}")
    print(f"  Corrupt:       {summary['corrupt_images']}")
    print(f"  Invalid labels:{summary['invalid_labels']}")
    print(f"  Empty labels:  {summary['empty_labels']}")
    print(f"  Tiny bboxes:   {summary['tiny_bboxes']}")
    print(f"  Annotations:   {summary['total_annotations']}")

    print(f"\n  Class distribution:")
    for cls_id in sorted(CLASS_NAMES.keys()):
        count = summary["class_counts"].get(cls_id, 0)
        pct = count / max(summary["total_annotations"], 1) * 100
        bar = "#" * int(pct / 2)
        print(f"    {CLASS_NAMES[cls_id]:15s}: {count:6d} ({pct:5.1f}%) {bar}")

    if summary["image_sizes"]:
        widths = [s[0] for s in summary["image_sizes"]]
        heights = [s[1] for s in summary["image_sizes"]]
        print(f"\n  Image sizes:")
        print(f"    Width:  min={min(widths)}, max={max(widths)}, "
              f"mean={np.mean(widths):.0f}")
        print(f"    Height: min={min(heights)}, max={max(heights)}, "
              f"mean={np.mean(heights):.0f}")

    if summary["errors"]:
        print(f"\n  ERRORS ({len(summary['errors'])}):")
        for err in summary["errors"][:10]:
            print(f"    [ERROR] {err}")
        if len(summary["errors"]) > 10:
            print(f"    ... and {len(summary['errors']) - 10} more")

    if summary["warnings"]:
        print(f"\n  WARNINGS ({len(summary['warnings'])}):")
        for warn in summary["warnings"][:10]:
            print(f"    [WARN] {warn}")

    return summary


def verify_dataset(data_dir: str):
    """Main verification function."""
    print(f"Verifying dataset: {data_dir}")

    # Check data.yaml exists
    yaml_path = os.path.join(data_dir, "data.yaml")
    if os.path.exists(yaml_path):
        with open(yaml_path, "r") as f:
            import yaml
            config = yaml.safe_load(f)
        print(f"\ndata.yaml found:")
        print(f"  Classes: {config.get('names', 'N/A')}")
        print(f"  NC:      {config.get('nc', 'N/A')}")
    else:
        print(f"\nWARNING: data.yaml not found at {yaml_path}")

    # Verify each split
    all_summaries = {}
    for split in ["train", "val", "test"]:
        split_dir = os.path.join(data_dir, split)
        if os.path.isdir(split_dir):
            all_summaries[split] = verify_split(split, split_dir)
        else:
            print(f"\n  {split}/ not found — skipping")

    # Overall summary
    print(f"\n{'='*60}")
    print(f"  OVERALL SUMMARY")
    print(f"{'='*60}")

    total_images = sum(s.get("total_images", 0) for s in all_summaries.values())
    total_errors = sum(len(s.get("errors", [])) for s in all_summaries.values())
    total_corrupt = sum(s.get("corrupt_images", 0) for s in all_summaries.values())

    print(f"\n  Total images across all splits: {total_images}")

    # Split ratios
    if total_images > 0:
        print(f"\n  Split ratios:")
        for split, summary in all_summaries.items():
            count = summary.get("total_images", 0)
            pct = count / total_images * 100
            print(f"    {split:6s}: {count:5d} ({pct:.1f}%)")

    # Overall class balance
    print(f"\n  Overall class distribution:")
    total_class_counts = defaultdict(int)
    for summary in all_summaries.values():
        for cls_id, count in summary.get("class_counts", {}).items():
            total_class_counts[cls_id] += count
    total_annot = sum(total_class_counts.values())
    for cls_id in sorted(CLASS_NAMES.keys()):
        count = total_class_counts.get(cls_id, 0)
        pct = count / max(total_annot, 1) * 100
        print(f"    {CLASS_NAMES[cls_id]:15s}: {count:6d} ({pct:.1f}%)")

    # Final verdict
    if total_errors == 0 and total_corrupt == 0:
        print(f"\n  [PASS] Dataset looks clean. Ready for training.")
    else:
        print(f"\n  [FAIL] ISSUES FOUND — {total_corrupt} corrupt images, "
              f"{total_errors} errors. Fix before training.")


def main():
    parser = argparse.ArgumentParser(
        description="Verify dataset integrity before training"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        # required=True,
        default="dev/data",
        help="Path to  directory with train/val/test splits",
    )
    args = parser.parse_args()
    verify_dataset(args.data_dir)


if __name__ == "__main__":
    main()
