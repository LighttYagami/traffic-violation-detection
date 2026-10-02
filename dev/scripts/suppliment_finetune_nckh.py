# Copy ~500 images from original train set that contain no-helmet labels
from pathlib import Path
import shutil

src_images = Path(r"C:\Users\rajcr\Desktop\Study\MTech\Courses\CV\Project\Dataset\leaked_data_split\train\images")
src_labels = Path(r"C:\Users\rajcr\Desktop\Study\MTech\Courses\CV\Project\Dataset\leaked_data_split\train\labels")
dst_images = Path(r"C:\Users\rajcr\Desktop\Study\MTech\Courses\CV\Project\Dataset\finetune_nckh_data\train\images")
dst_labels= Path(r"C:\Users\rajcr\Desktop\Study\MTech\Courses\CV\Project\Dataset\finetune_nckh_data\train\labels")

count = 0
for lbl in src_labels.glob("*.txt"):
    has_nohelmet = any(line.strip().split()[0] == "2" for line in open(lbl))
    if has_nohelmet:
        shutil.copy2(lbl, dst_labels / lbl.name)
        img = src_images / lbl.stem
        for ext in [".jpg", ".jpeg", ".png"]:
            src_img = src_images / f"{lbl.stem}{ext}"
            if src_img.exists():
                shutil.copy2(src_img, dst_images / src_img.name)
                count += 1
                break
    if count >= 500:
        break

print(f"Added {count} images with no-helmet annotations")