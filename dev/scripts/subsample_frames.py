"""
subsample_frames.py
====================
Subsamples the same-camera video frames (prefix "frame_") to prevent
viewpoint overfitting. Moves excess frames + their labels to an archive folder.

Usage:
    python subsample_frames.py \
        --data_dir /path/to/data/train \
        --keep 1500 \
        --seed 42

Before running:
    data/train/
    ├── images/   (8,466 .jpg files including ~3,500 frame_* files)
    └── labels/   (8,466 .txt files)

After running:
    data/train/
    ├── images/   (~6,466 .jpg files, only ~1,500 frame_* remain)
    └── labels/   (matching .txt files)
    data/archived_frames/
    ├── images/   (~2,000 removed frame_* .jpg files)
    └── labels/   (~2,000 removed frame_* .txt files)
"""

import os
import shutil
import random
import argparse
from pathlib import Path


def find_frame_images(images_dir: str, prefix: str = "frame_") -> list:
    """Find all images with the given prefix."""
    frame_files = []
    for f in os.listdir(images_dir):
        if f.lower().startswith(prefix) and f.lower().endswith(".jpg"):
            frame_files.append(f)
    return sorted(frame_files)


def subsample_and_archive(
    data_dir: str,
    keep: int = 1500,
    seed: int = 42,
    prefix: str = "frame_",
    dry_run: bool = False,
):
    """
    Subsample frame_* images, move excess to archive.
    
    Args:
        data_dir: Path to train/ directory containing images/ and labels/
        keep: Number of frame images to keep
        seed: Random seed for reproducibility
        prefix: Filename prefix to identify same-camera frames
        dry_run: If True, print what would be done without moving files
    """
    images_dir = os.path.join(data_dir, "images")
    labels_dir = os.path.join(data_dir, "labels")

    # Validate directories exist
    if not os.path.isdir(images_dir):
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
    if not os.path.isdir(labels_dir):
        raise FileNotFoundError(f"Labels directory not found: {labels_dir}")

    # Find all frame images
    frame_files = find_frame_images(images_dir, prefix)
    total_frames = len(frame_files)
    print(f"Found {total_frames} images with prefix '{prefix}'")

    if total_frames <= keep:
        print(f"Total frames ({total_frames}) <= keep ({keep}). Nothing to do.")
        return

    # Random sample to keep
    random.seed(seed)
    keep_files = set(random.sample(frame_files, keep))
    remove_files = [f for f in frame_files if f not in keep_files]

    print(f"Keeping:  {len(keep_files)}")
    print(f"Removing: {len(remove_files)}")

    if dry_run:
        print("\n[DRY RUN] Would remove these files:")
        for f in remove_files[:10]:
            print(f"  {f}")
        if len(remove_files) > 10:
            print(f"  ... and {len(remove_files) - 10} more")
        return

    # Create archive directory (sibling to data_dir)
    archive_dir = os.path.join(os.path.dirname(data_dir), "archived_frames")
    archive_images = os.path.join(archive_dir, "images")
    archive_labels = os.path.join(archive_dir, "labels")
    os.makedirs(archive_images, exist_ok=True)
    os.makedirs(archive_labels, exist_ok=True)

    # Move excess frames to archive
    moved_images = 0
    moved_labels = 0
    missing_labels = 0

    for img_file in remove_files:
        # Move image
        src_img = os.path.join(images_dir, img_file)
        dst_img = os.path.join(archive_images, img_file)
        shutil.move(src_img, dst_img)
        moved_images += 1

        # Move corresponding label
        label_file = os.path.splitext(img_file)[0] + ".txt"
        src_label = os.path.join(labels_dir, label_file)
        dst_label = os.path.join(archive_labels, label_file)

        if os.path.exists(src_label):
            shutil.move(src_label, dst_label)
            moved_labels += 1
        else:
            missing_labels += 1

    # Summary
    print(f"\nDone!")
    print(f"  Moved {moved_images} images to {archive_images}")
    print(f"  Moved {moved_labels} labels to {archive_labels}")
    if missing_labels > 0:
        print(f"  WARNING: {missing_labels} images had no matching label file")

    # Count remaining
    remaining_total = len(os.listdir(images_dir))
    remaining_frames = len(find_frame_images(images_dir, prefix))
    print(f"\nRemaining in {images_dir}:")
    print(f"  Total images: {remaining_total}")
    print(f"  Frame images: {remaining_frames}")
    print(f"  Other images: {remaining_total - remaining_frames}")


def main():
    parser = argparse.ArgumentParser(
        description="Subsample same-camera video frames to prevent viewpoint overfitting"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        # required=True,
        default=r"C:\Users\rajcr\Desktop\Study\MTech\Courses\CV\Project\Dataset\MainDataset\train",
        help="Path to train/ directory containing images/ and labels/",
    )
    parser.add_argument(
        "--keep",
        type=int,
        default=1500,
        help="Number of frame images to keep (default: 1500)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="frame_",
        help="Filename prefix to identify same-camera frames (default: frame_)",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print what would be done without moving files",
    )

    args = parser.parse_args()
    subsample_and_archive(
        data_dir=args.data_dir,
        keep=args.keep,
        seed=args.seed,
        prefix=args.prefix,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
