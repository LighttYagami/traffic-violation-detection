"""
YOLO26s Fine-tuning on Kaggle T4 — Bike Detection Improvement
===============================================================

Kaggle Notebook Setup:
1. Upload your best.pt as a Kaggle Dataset (e.g., "yolo26s-checkpoint")
2. Upload your fine-tuning images+labels as a Kaggle Dataset (e.g., "bike-finetune-data")
3. Enable GPU T4 x2 in notebook settings
4. Paste this entire script into a single cell and run

Directory structure for your uploaded dataset:
    bike-finetune-data/
        train/
            images/   ← .jpg/.png files
            labels/   ← .txt YOLO format
        val/
            images/
            labels/

T4 Constraints:
- 16 GB VRAM → batch=8 at imgsz=1280 (safe), batch=16 may work
- 12 hour session limit → keep epochs ≤ 20
- Save outputs to /kaggle/working/ for download
"""

# ============================================================
# CELL 1: Install & Setup
# ============================================================
import subprocess
import sys

subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "ultralytics"])

import os
import torch
import shutil
from pathlib import Path
from ultralytics import YOLO

# Verify GPU
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB")

# ============================================================
# CELL 2: Configuration — EDIT THESE PATHS
# ============================================================

# >>> EDIT THESE to match your Kaggle dataset paths <<<
WEIGHTS_PATH = "/kaggle/input/yolo26s-checkpoint/best.pt"      # your uploaded best.pt
DATA_ROOT = "/kaggle/input/bike-finetune-data"                  # your uploaded dataset

# Training config (tuned for T4)
CONFIG = {
    "imgsz": 1280,
    "batch": 8,             # safe for T4 16GB; try 12 if no OOM
    "epochs": 15,           # enough for fine-tuning, fits in 12hr limit
    "patience": 5,          # early stop if no improvement for 5 epochs
    "freeze": 10,           # freeze backbone, train neck + head only
    "lr0": 0.001,           # conservative — preserve learned features
    "lrf": 0.1,             # final LR = lr0 * 0.1
    "weight_decay": 0.0005,
    "dropout": 0.1,         # light regularization for small dataset
    "amp": True,            # mixed precision — critical for T4 memory
}

OUTPUT_DIR = "/kaggle/working"

# ============================================================
# CELL 3: Create dataset YAML
# ============================================================

# Verify dataset exists
data_root = Path(DATA_ROOT)
for split in ["train", "val"]:
    img_dir = data_root / split / "images"
    lbl_dir = data_root / split / "labels"
    if img_dir.exists():
        n_imgs = len(list(img_dir.glob("*.[jJpP][pPnN][gG]*")))
        n_lbls = len(list(lbl_dir.glob("*.txt")))
        print(f"  {split}: {n_imgs} images, {n_lbls} labels")
    else:
        print(f"  WARNING: {img_dir} not found!")

# Write YAML (Kaggle input is read-only, so write to working dir)
yaml_path = Path(OUTPUT_DIR) / "finetune_data.yaml"
yaml_content = f"""
path: {DATA_ROOT}
train: train/images
val: val/images

nc: 4
names:
  0: bike
  1: helmet
  2: no-helmet
  3: number-plate
""".strip()

yaml_path.write_text(yaml_content)
print(f"\nDataset YAML written to: {yaml_path}")

# ============================================================
# CELL 4: Verify checkpoint loads
# ============================================================

weights = Path(WEIGHTS_PATH)
if not weights.exists():
    raise FileNotFoundError(
        f"Checkpoint not found at {WEIGHTS_PATH}\n"
        f"Available files in /kaggle/input/:\n"
        f"{os.listdir('/kaggle/input/')}"
    )

model = YOLO(str(weights))
print(f"Model loaded: {weights.name}")
print(f"Model type: {model.type}")

# Quick sanity check — run val on the fine-tune val set before training
print("\n--- Pre-training baseline on fine-tune val set ---")
baseline = model.val(data=str(yaml_path), imgsz=CONFIG["imgsz"])
print(f"Baseline mAP@50: {baseline.box.map50:.4f}")
class_names = ["bike", "helmet", "no-helmet", "number-plate"]
for i, name in enumerate(class_names):
    if i < len(baseline.box.ap50):
        print(f"  {name} AP@50: {baseline.box.ap50[i]:.4f}")

# ============================================================
# CELL 5: Fine-tune
# ============================================================

print(f"\n{'='*60}")
print(f"  Starting fine-tuning on Kaggle T4")
print(f"{'='*60}")
for k, v in CONFIG.items():
    print(f"  {k}: {v}")
print(f"{'='*60}\n")

results = model.train(
    data=str(yaml_path),
    epochs=CONFIG["epochs"],
    imgsz=CONFIG["imgsz"],
    batch=CONFIG["batch"],
    patience=CONFIG["patience"],
    amp=CONFIG["amp"],

    # Learning rate
    lr0=CONFIG["lr0"],
    lrf=CONFIG["lrf"],
    warmup_epochs=2,           # short warmup since we're fine-tuning
    warmup_bias_lr=0.01,       # lower than default 0.1

    # Regularization
    weight_decay=CONFIG["weight_decay"],
    dropout=CONFIG["dropout"],

    # Freeze backbone
    freeze=CONFIG["freeze"],

    # Augmentation — moderate for small dataset
    hsv_h=0.015,
    hsv_s=0.5,
    hsv_v=0.3,
    scale=0.3,
    fliplr=0.5,
    mosaic=1.0,
    flipud=0.0,
    degrees=0.0,
    shear=0.0,

    # Output
    project=f"{OUTPUT_DIR}/runs",
    name="yolo26s_bike_finetune",
    save=True,
    save_period=5,             # checkpoint every 5 epochs
    plots=True,
    verbose=True,
    device=0,
)

# ============================================================
# CELL 6: Evaluate & Compare
# ============================================================

best_pt = Path(OUTPUT_DIR) / "runs" / "yolo26s_bike_finetune" / "weights" / "best.pt"
last_pt = Path(OUTPUT_DIR) / "runs" / "yolo26s_bike_finetune" / "weights" / "last.pt"

if best_pt.exists():
    print(f"\n{'='*60}")
    print(f"  Post-training Evaluation")
    print(f"{'='*60}\n")

    finetuned = YOLO(str(best_pt))
    metrics = finetuned.val(data=str(yaml_path), imgsz=CONFIG["imgsz"])

    print(f"\n  {'Metric':<20} {'Before':>10} {'After':>10} {'Delta':>10}")
    print(f"  {'-'*50}")
    print(f"  {'mAP@50':<20} {baseline.box.map50:>10.4f} {metrics.box.map50:>10.4f} {metrics.box.map50-baseline.box.map50:>+10.4f}")
    print(f"  {'mAP@50-95':<20} {baseline.box.map:>10.4f} {metrics.box.map:>10.4f} {metrics.box.map-baseline.box.map:>+10.4f}")

    for i, name in enumerate(class_names):
        if i < len(metrics.box.ap50) and i < len(baseline.box.ap50):
            before = baseline.box.ap50[i]
            after = metrics.box.ap50[i]
            print(f"  {name+' AP@50':<20} {before:>10.4f} {after:>10.4f} {after-before:>+10.4f}")

    print(f"\n  Best weights: {best_pt}")

    # Copy best.pt to easy download location
    download_path = Path(OUTPUT_DIR) / "best_finetuned.pt"
    shutil.copy2(best_pt, download_path)
    print(f"  Copied to: {download_path}  ← download this!")

else:
    print(f"WARNING: best.pt not found at {best_pt}")
    if last_pt.exists():
        print(f"  last.pt available at: {last_pt}")

# ============================================================
# CELL 7: Quick inference test (optional)
# ============================================================

# Uncomment to test on a sample image
# test_img = "/kaggle/input/bike-finetune-data/val/images/some_image.jpg"
# if Path(test_img).exists():
#     results = finetuned.predict(test_img, imgsz=1280, conf=0.25)
#     for r in results:
#         print(f"Detections: {len(r.boxes)}")
#         for box in r.boxes:
#             cls = int(box.cls[0])
#             conf = float(box.conf[0])
#             print(f"  {class_names[cls]}: {conf:.3f}")

print("\n✅ Done! Download best_finetuned.pt from /kaggle/working/")
