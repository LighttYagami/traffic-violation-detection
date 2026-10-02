"""
Model Download Script
=====================
Downloads all required model weights from HuggingFace into the models/ directory.
Run this once before evaluation.

Usage:
    python download_models.py
"""

import os
import sys
import subprocess

# ============================================================
# CONFIGURATION — Update this after uploading models
# ============================================================
HF_REPO = "YOUR_USERNAME/traffic-violation-models"
# ============================================================

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")


def main():
    # Check if models already exist
    yolo_path = os.path.join(MODELS_DIR, "helmet_detector.pt")
    if os.path.exists(yolo_path):
        print(f"Models already exist at {MODELS_DIR}")
        resp = input("Re-download? (y/n): ").strip().lower()
        if resp != "y":
            print("Skipping download.")
            return

    # Install huggingface_hub if needed
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("Installing huggingface_hub...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "huggingface_hub", "-q"])
        from huggingface_hub import snapshot_download

    os.makedirs(MODELS_DIR, exist_ok=True)

    print(f"Downloading models from HuggingFace ({HF_REPO})...")
    snapshot_download(repo_id=HF_REPO, local_dir=MODELS_DIR)

    # Verify
    print("\nVerifying downloaded models...")
    expected = [
        "helmet_detector.pt",
        "paddleocr/PP-OCRv5_server_det/inference.pdiparams",
        "paddleocr/en_PP-OCRv5_mobile_rec/inference.pdiparams",
        "paddleocr/PP-LCNet_x1_0_textline_ori/inference.pdiparams",
    ]

    all_ok = True
    for rel_path in expected:
        full_path = os.path.join(MODELS_DIR, rel_path)
        if os.path.exists(full_path):
            size_mb = os.path.getsize(full_path) / 1024 / 1024
            print(f"  OK  {rel_path} ({size_mb:.1f} MB)")
        else:
            print(f"  MISSING  {rel_path}")
            all_ok = False

    # Check total size
    total = 0
    for root, dirs, files in os.walk(MODELS_DIR):
        for f in files:
            total += os.path.getsize(os.path.join(root, f))
    total_mb = total / 1024 / 1024
    print(f"\nTotal models size: {total_mb:.1f} MB (limit: 250 MB)")

    if total_mb > 250:
        print("WARNING: Total size exceeds 250 MB limit!")

    if all_ok:
        print("\nAll models downloaded successfully. Ready for evaluation.")
    else:
        print("\nSome models are missing. Check the HuggingFace repo and try again.")


if __name__ == "__main__":
    main()
