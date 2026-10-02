"""
Remap NCKH 2023 Dataset for Fine-tuning
=========================================

NCKH classes → Our classes:
  - motorcyclist → bike (0)
  - helmet → helmet (1)
  - (no equivalent) → no-helmet (2)  ← won't exist in NCKH
  - license_plate → number-plate (3)

This script:
1. Reads NCKH label files
2. Remaps class IDs to match our dataset
3. Copies images + remapped labels into a fine-tune training folder
4. Creates a dataset YAML that trains on NCKH but validates on your original val set

Usage:
    python remap_nckh.py \
        --nckh-root /path/to/nckh_dataset \
        --original-val /path/to/original/dataset/val \
        --output ./finetune_nckh_data

Then in kaggle_finetune_yolo26s.py, point DATA_ROOT to the output folder.
"""

import argparse
import shutil
import os
import yaml
from pathlib import Path
from collections import Counter


def detect_nckh_class_mapping(data_yaml_path):
    """
    Auto-detect NCKH class names from its data.yaml.
    Returns a dict mapping NCKH class_id → our class_id.
    """
    if data_yaml_path and Path(data_yaml_path).exists():
        with open(data_yaml_path) as f:
            data = yaml.safe_load(f)
        names = data.get("names", {})
        # Could be a list or dict
        if isinstance(names, list):
            names = {i: n for i, n in enumerate(names)}
        print(f"  NCKH classes from YAML: {names}")
        return build_mapping(names)

    return None


def build_mapping(nckh_names):
    """
    Map NCKH class names to our class IDs.
    Our classes: 0=bike, 1=helmet, 2=no-helmet, 3=number-plate
    """
    # Normalize: lowercase, strip whitespace
    normalized = {k: v.lower().strip().replace("_", " ") for k, v in nckh_names.items()}

    mapping = {}
    for nckh_id, name in normalized.items():
        if name in ["motorcyclist", "motorcycle", "bike", "motorbike", "two-wheeler", "two wheeler"]:
            mapping[int(nckh_id)] = 0  # bike
        elif name in ["helmet", "with helmet", "with-helmet"]:
            mapping[int(nckh_id)] = 1  # helmet
        elif name in ["no helmet", "no-helmet", "without helmet", "without-helmet"]:
            mapping[int(nckh_id)] = 2  # no-helmet
        elif name in ["license plate", "license-plate", "licence", "licence plate",
                       "number plate", "number-plate", "numberplate", "plate"]:
            mapping[int(nckh_id)] = 3  # number-plate
        else:
            print(f"  WARNING: Unknown NCKH class '{name}' (id={nckh_id}) — SKIPPING")

    return mapping


def remap_label_file(src_path, dst_path, class_mapping):
    """
    Read a YOLO label file, remap class IDs, write to dst.
    Lines with unmapped classes are dropped.
    Returns count of kept and dropped annotations.
    """
    kept = 0
    dropped = 0
    lines_out = []

    with open(src_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            old_cls = int(parts[0])
            if old_cls in class_mapping:
                new_cls = class_mapping[old_cls]
                parts[0] = str(new_cls)
                lines_out.append(" ".join(parts))
                kept += 1
            else:
                dropped += 1

    with open(dst_path, "w") as f:
        f.write("\n".join(lines_out) + "\n" if lines_out else "")

    return kept, dropped


def find_images_and_labels(root):
    """Find all image-label pairs in a dataset (handles various structures)."""
    root = Path(root)
    pairs = []

    # Try: root/images/ + root/labels/
    # Try: root/train/images/ + root/train/labels/
    # Try: root/ (flat)
    search_dirs = [
        (root / "images", root / "labels"),
        (root / "train" / "images", root / "train" / "labels"),
        (root / "valid" / "images", root / "valid" / "labels"),
        (root / "val" / "images", root / "val" / "labels"),
        (root / "test" / "images", root / "test" / "labels"),
        (root, root),
    ]

    seen = set()
    for img_dir, lbl_dir in search_dirs:
        if not img_dir.exists():
            continue
        for img_path in img_dir.iterdir():
            if img_path.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]:
                label_path = lbl_dir / (img_path.stem + ".txt")
                if label_path.exists() and img_path.stem not in seen:
                    seen.add(img_path.stem)
                    pairs.append((img_path, label_path))

    return pairs


def main():
    parser = argparse.ArgumentParser(description="Remap NCKH dataset for fine-tuning")
    parser.add_argument("--nckh-root", type=str, 
                        default=r"C:\Users\rajcr\Desktop\Study\MTech\Courses\CV\Project\Dataset\NCHK",
                        help="Path to NCKH dataset root")
    parser.add_argument("--nckh-yaml", type=str, 
                        default=r"C:\Users\rajcr\Desktop\Study\MTech\Courses\CV\Project\Dataset\NCHK\data.yaml",
                        help="Path to NCKH data.yaml (auto-detected if not specified)")
    parser.add_argument("--original-val", type=str, 
                        default=r"C:\Users\rajcr\Desktop\Study\MTech\Courses\CV\Project\Dataset\leaked_data_split\val",
                        help="Path to your original dataset val/ folder (with images/ and labels/)")
    parser.add_argument("--output", type=str, 
                        default=r"C:\Users\rajcr\Desktop\Study\MTech\Courses\CV\Project\Dataset\finetune_nckh_data",
                        help="Output directory for remapped dataset")
    parser.add_argument("--class-mapping", type=str, default='0:1,1:3,2:0',
                        help="Manual mapping as 'nckh_id:our_id,...' e.g. '0:0,1:1,2:3'")
    args = parser.parse_args()

    nckh_root = Path(args.nckh_root)
    output = Path(args.output)

    # ---- Step 1: Determine class mapping ----
    print("\n[1/4] Determining class mapping...")

    class_mapping = None

    # Manual override
    if args.class_mapping:
        class_mapping = {}
        for pair in args.class_mapping.split(","):
            src, dst = pair.split(":")
            class_mapping[int(src)] = int(dst)
        print(f"  Manual mapping: {class_mapping}")

    # Auto-detect from YAML
    if class_mapping is None:
        yaml_candidates = [
            args.nckh_yaml,
            nckh_root / "data.yaml",
            nckh_root / "dataset.yaml",
            nckh_root / "README.dataset.yaml",
        ]
        for yp in yaml_candidates:
            if yp and Path(yp).exists():
                class_mapping = detect_nckh_class_mapping(str(yp))
                if class_mapping:
                    break

    # Fallback: try to infer from label contents
    if class_mapping is None:
        print("  WARNING: Could not auto-detect classes.")
        print("  Assuming NCKH classes: 0=motorcyclist, 1=helmet, 2=license_plate")
        print("  If wrong, re-run with --nckh-yaml or --class-mapping")
        class_mapping = {0: 0, 1: 1, 2: 3}  # motorcyclist→bike, helmet→helmet, license_plate→number-plate

    our_names = {0: "bike", 1: "helmet", 2: "no-helmet", 3: "number-plate"}
    print(f"\n  Final mapping:")
    for src_id, dst_id in sorted(class_mapping.items()):
        print(f"    NCKH class {src_id} → {our_names[dst_id]} ({dst_id})")

    # ---- Step 2: Find and remap NCKH data ----
    print("\n[2/4] Finding NCKH images and labels...")
    pairs = find_images_and_labels(nckh_root)
    print(f"  Found {len(pairs)} image-label pairs")

    if not pairs:
        print("\n  ERROR: No image-label pairs found!")
        print(f"  Searched in: {nckh_root}")
        print(f"  Expected structure: images/ + labels/ subdirectories")
        return

    # Create output dirs
    train_img_dir = output / "train" / "images"
    train_lbl_dir = output / "train" / "labels"
    train_img_dir.mkdir(parents=True, exist_ok=True)
    train_lbl_dir.mkdir(parents=True, exist_ok=True)

    print("\n[3/4] Remapping labels and copying files...")
    total_kept = 0
    total_dropped = 0
    new_class_counts = Counter()

    for img_path, lbl_path in pairs:
        # Copy image
        dst_img = train_img_dir / img_path.name
        shutil.copy2(img_path, dst_img)

        # Remap and write label
        dst_lbl = train_lbl_dir / lbl_path.name
        kept, dropped = remap_label_file(lbl_path, dst_lbl, class_mapping)
        total_kept += kept
        total_dropped += dropped

        # Count new classes
        with open(dst_lbl) as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    new_class_counts[int(parts[0])] += 1

    print(f"  Annotations kept:    {total_kept}")
    print(f"  Annotations dropped: {total_dropped}")
    print(f"\n  Remapped class distribution:")
    for cls_id in sorted(new_class_counts):
        print(f"    {our_names.get(cls_id, f'unknown-{cls_id}')}: {new_class_counts[cls_id]}")

    # ---- Step 3: Set up val as symlink/copy to original val ----
    print("\n[4/4] Setting up validation from original dataset...")
    original_val = Path(args.original_val)
    val_img_src = original_val / "images"
    val_lbl_src = original_val / "labels"

    if not val_img_src.exists():
        # Maybe it's the parent and images/labels are directly inside
        print(f"  WARNING: {val_img_src} not found")
        print(f"  Trying {original_val} directly...")
        val_img_src = original_val
        val_lbl_src = original_val

    val_img_dst = output / "val" / "images"
    val_lbl_dst = output / "val" / "labels"

    # Try symlink first (faster, saves space), fall back to copy
    try:
        val_img_dst.parent.mkdir(parents=True, exist_ok=True)
        if val_img_dst.exists() or val_img_dst.is_symlink():
            if val_img_dst.is_symlink():
                val_img_dst.unlink()
            else:
                shutil.rmtree(val_img_dst)
        if val_lbl_dst.exists() or val_lbl_dst.is_symlink():
            if val_lbl_dst.is_symlink():
                val_lbl_dst.unlink()
            else:
                shutil.rmtree(val_lbl_dst)

        os.symlink(val_img_src.resolve(), val_img_dst)
        os.symlink(val_lbl_src.resolve(), val_lbl_dst)
        print(f"  Symlinked val/images → {val_img_src}")
        print(f"  Symlinked val/labels → {val_lbl_src}")
    except OSError:
        # Symlinks not supported (e.g., Kaggle), copy instead
        print(f"  Symlink failed, copying val set...")
        shutil.copytree(val_img_src, val_img_dst, dirs_exist_ok=True)
        shutil.copytree(val_lbl_src, val_lbl_dst, dirs_exist_ok=True)
        print(f"  Copied val set to {output / 'val'}")

    # Count val set
    n_val_imgs = len(list(val_img_dst.glob("*.[jJpP][pPnN][gG]*"))) if val_img_dst.is_dir() else 0
    print(f"  Val set: {n_val_imgs} images")

    # ---- Step 4: Write dataset YAML ----
    yaml_path = output / "data.yaml"
    yaml_content = {
        "path": str(output.resolve()),
        "train": "train/images",
        "val": "val/images",
        "nc": 4,
        "names": {0: "bike", 1: "helmet", 2: "no-helmet", 3: "number-plate"},
    }
    with open(yaml_path, "w") as f:
        yaml.dump(yaml_content, f, default_flow_style=False)

    # ---- Summary ----
    print(f"\n{'='*60}")
    print(f"  Dataset ready for fine-tuning!")
    print(f"{'='*60}")
    print(f"  Train: {len(pairs)} NCKH images (remapped)")
    print(f"  Val:   {n_val_imgs} images (original val set)")
    print(f"  YAML:  {yaml_path}")
    print(f"")
    print(f"  Next step:")
    print(f"    python kaggle_finetune_yolo26s.py")
    print(f"    Set DATA_ROOT = \"{output.resolve()}\"")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
