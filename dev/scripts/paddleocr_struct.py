import os

for folder in ["~/.paddleocr", "~/.paddlex"]:
    path = os.path.expanduser(folder)
    if os.path.exists(path):
        print(f"\n=== {path} ===")
        for root, dirs, files in os.walk(path):
            for f in files:
                full = os.path.join(root, f)
                size = os.path.getsize(full) / 1024 / 1024
                print(f"  {size:.1f}MB  {full}")
    else:
        print(f"{path} — not found")