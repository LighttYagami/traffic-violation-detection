"""
build_groundtruth_dataset.py
=============================
Scans ground_truth.json and copies the corresponding images + labels from a
source dataset into the GroundTruthData images/ and labels/ folders.

Already-present images are skipped automatically (idempotent).

Optional label remapping: set APPLY_LABEL_MAPPING = True and fill in
LABEL_MAPPING to translate source class IDs to your target class IDs.

Usage:
    python dev/scripts/build_groundtruth_dataset.py \
        --gt_json   "C:/path/to/GroundTruthData/ground_truth.json" \
        --src_images "C:/path/to/SourceDataset/train/images" \
        --src_labels "C:/path/to/SourceDataset/train/labels"
"""

import argparse
import json
import os
import shutil

# ── Label mapping config ──────────────────────────────────────────────────────
# Set to True to remap label class IDs from source to target format.
APPLY_LABEL_MAPPING = False

# Maps source class ID → target class ID.
# LABEL_MAPPING = {src_id: dst_id, ...}
# Example below: NCHK (helmet=0, license_plate=1, motorcyclist=2)
#             → MainDataset (bike=0, helmet=1, no-helmet=2, number-plate=3)
LABEL_MAPPING: dict[int, int] = {
    0: 1,   # helmet       → helmet
    1: 3,   # license_plate → number-plate
    2: 0,   # motorcyclist  → bike
}
# ─────────────────────────────────────────────────────────────────────────────

def remap_label_file(src_path: str, dst_path: str, mapping: dict[int, int]) -> None:
    """Read a YOLO label file, remap class IDs, write to dst_path."""
    lines_out = []
    with open(src_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            src_cls = int(parts[0])
            dst_cls = mapping.get(src_cls, src_cls)  # pass-through if not in map
            lines_out.append(f"{dst_cls} {' '.join(parts[1:])}")
    with open(dst_path, "w") as f:
        f.write("\n".join(lines_out) + "\n" if lines_out else "")


def build_groundtruth(gt_json: str, src_images: str, src_labels: str) -> None:
    gt_json = os.path.abspath(gt_json)
    src_images = os.path.abspath(src_images)
    src_labels = os.path.abspath(src_labels)

    gt_dir = os.path.dirname(gt_json)
    dst_images = os.path.join(gt_dir, "images")
    dst_labels = os.path.join(gt_dir, "labels")
    os.makedirs(dst_images, exist_ok=True)
    os.makedirs(dst_labels, exist_ok=True)

    with open(gt_json, "r") as f:
        ground_truth: dict = json.load(f)

    total = len(ground_truth)
    skipped = 0
    copied = 0
    missing = 0

    print(f"Ground truth JSON : {gt_json}")
    print(f"Source images     : {src_images}")
    print(f"Source labels     : {src_labels}")
    print(f"Destination images: {dst_images}")
    print(f"Destination labels: {dst_labels}")
    print(f"Label mapping     : {'ON' if APPLY_LABEL_MAPPING else 'OFF'}")
    if APPLY_LABEL_MAPPING:
        print(f"  Mapping: {LABEL_MAPPING}")
    print(f"\nProcessing {total} entries...\n")

    for img_name in ground_truth:
        stem = os.path.splitext(img_name)[0]
        dst_img_path = os.path.join(dst_images, img_name)

        # Skip if already exists
        if os.path.exists(dst_img_path):
            skipped += 1
            continue

        # Find image in source
        src_img_path = os.path.join(src_images, img_name)
        if not os.path.exists(src_img_path):
            print(f"  [MISSING] Image not found: {img_name}")
            missing += 1
            continue

        # Copy image
        shutil.copy2(src_img_path, dst_img_path)

        # Copy / remap label
        label_name = stem + ".txt"
        src_lbl_path = os.path.join(src_labels, label_name)
        dst_lbl_path = os.path.join(dst_labels, label_name)

        if os.path.exists(src_lbl_path):
            if APPLY_LABEL_MAPPING:
                remap_label_file(src_lbl_path, dst_lbl_path, LABEL_MAPPING)
            else:
                shutil.copy2(src_lbl_path, dst_lbl_path)
        else:
            # Write empty label file so the dataset is consistent
            open(dst_lbl_path, "w").close()
            print(f"  [WARN] Label not found for {img_name} — created empty label")

        copied += 1

    print(f"\nDone.")
    print(f"  Copied : {copied}")
    print(f"  Skipped: {skipped} (already existed)")
    print(f"  Missing: {missing} (image not found in source)")


def main():
    parser = argparse.ArgumentParser(
        description="Copy ground-truth images and labels from source dataset"
    )
    parser.add_argument(
        "--gt_json",
        default=r"C:\Users\rajcr\Desktop\Study\MTech\Courses\CV\Project\Dataset\GroundTruthData\ground_truth.json",
        help="Path to ground_truth.json",
    )
    parser.add_argument(
        "--src_images",
        default=r"C:\Users\rajcr\Desktop\Study\MTech\Courses\CV\Project\Dataset\leaked_data_split\test\images",
        help="Path to source images directory",
    )
    parser.add_argument(
        "--src_labels",
        default=r"C:\Users\rajcr\Desktop\Study\MTech\Courses\CV\Project\Dataset\leaked_data_split\test\labels",
        help="Path to source labels directory",
    )
    args = parser.parse_args()
    build_groundtruth(args.gt_json, args.src_images, args.src_labels)


if __name__ == "__main__":
    main()
