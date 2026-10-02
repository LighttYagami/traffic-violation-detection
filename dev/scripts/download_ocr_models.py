"""
download_ocr_models.py
=======================
Downloads PaddleOCR and EasyOCR models to their default cache directories.
Run this once before export_model.py.

  PaddleOCR  → ~/.paddleocr/
  EasyOCR    → ~/.EasyOCR/

Usage:
    python dev/scripts/download_ocr_models.py
"""

import os
import torch

# Disable OneDNN and PIR executor to avoid attribute conversion errors on Windows
os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["FLAGS_enable_pir_api"] = "0"

gpu = torch.cuda.is_available()
print(f"Device: {'GPU' if gpu else 'CPU'}")

# ── PaddleOCR ────────────────────────────────────────────────────────────────
print("\n[1/2] Downloading PaddleOCR models...")
try:
    from paddleocr import PaddleOCR
    import pathlib
    ocr = PaddleOCR(
        use_textline_orientation=True,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        lang="en",
    )
    # Run inference to trigger lazy model download.
    # The OneDNN/PIR runtime error on Windows fires *after* the models are
    # already saved to disk, so we catch it and verify the cache instead.
    import numpy as np
    dummy = np.zeros((64, 200, 3), dtype=np.uint8)
    try:
        ocr.ocr(dummy)
    except Exception:
        pass  # expected on Windows paddlepaddle with OneDNN

    # Locate the cache — PaddleOCR 3.x uses ~/.paddlex, older uses ~/.paddleocr
    home = pathlib.Path.home()
    cache_dirs = [home / ".paddlex", home / ".paddleocr"]
    found = [str(d) for d in cache_dirs if d.exists()]
    if found:
        print(f"  PaddleOCR models downloaded to: {found[0]}")
    else:
        raise RuntimeError("Model cache directory not found after inference call.")
except Exception as e:
    print(f"  PaddleOCR download failed: {e}")
    print("  Install with: pip install paddlepaddle paddleocr")

# ── EasyOCR ──────────────────────────────────────────────────────────────────
print("\n[2/2] Downloading EasyOCR models...")
try:
    import easyocr
    reader = easyocr.Reader(["en"], gpu=gpu)
    import numpy as np
    dummy = np.zeros((64, 200, 3), dtype=np.uint8)
    result = reader.readtext(dummy)
    print("  EasyOCR models downloaded and verified.")
except Exception as e:
    print(f"  EasyOCR download failed: {e}")
    print("  Install with: pip install easyocr")

print("\nDone. Now run export_model.py to copy models to submission/.")
