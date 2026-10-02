import shutil
from pathlib import Path

# Your original train set
orig_img = Path(r"C:\Users\rajcr\Desktop\Study\MTech\Courses\CV\Project\Dataset\leaked_data_split\train\images")
orig_lbl = Path(r"C:\Users\rajcr\Desktop\Study\MTech\Courses\CV\Project\Dataset\leaked_data_split\train\labels")

# NCKH remapped data
nckh_img = Path(r"C:\Users\rajcr\Desktop\Study\MTech\Courses\CV\Project\Dataset\finetune_nckh_data\train\images")
nckh_lbl = Path(r"C:\Users\rajcr\Desktop\Study\MTech\Courses\CV\Project\Dataset\finetune_nckh_data\train\labels")

# Combined output (Kaggle input is read-only, so copy to working)
out_img = Path("/kaggle/working/combined/train/images")
out_lbl = Path("/kaggle/working/combined/train/labels")
out_img.mkdir(parents=True, exist_ok=True)
out_lbl.mkdir(parents=True, exist_ok=True)

# Copy original
for f in orig_img.glob("*"): shutil.copy2(f, out_img / f.name)
for f in orig_lbl.glob("*"): shutil.copy2(f, out_lbl / f.name)

# Copy NCKH on top
for f in nckh_img.glob("*"): shutil.copy2(f, out_img / f.name)
for f in nckh_lbl.glob("*"): shutil.copy2(f, out_lbl / f.name)

# Symlink original val set
val_img = Path("/kaggle/working/combined/val/images")
val_lbl = Path("/kaggle/working/combined/val/labels")
val_img.parent.mkdir(parents=True, exist_ok=True)
shutil.copytree("/kaggle/input/original-dataset/val/images", val_img)
shutil.copytree("/kaggle/input/original-dataset/val/labels", val_lbl)

print(f"Train: {len(list(out_img.glob('*')))} images")
print(f"Val: {len(list(val_img.glob('*')))} images")