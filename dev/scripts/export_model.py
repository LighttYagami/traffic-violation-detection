"""
export_model.py
================
Copies best.pt and OCR models to submission directory.
Runs all pre-submission validation checks.

Usage:
    python export_model.py \
        --experiment_dir ../training_logs/exp_001_yolo26s_full_finetune \
        --submission_dir ../../submission/ROLL_NUMBER \
        --roll_number YOUR_ROLL_NUMBER
"""

import os
import sys
import re
import shutil
import time
import json
import argparse
from pathlib import Path


def get_dir_size_mb(path: str) -> float:
    """Get total size of directory in MB."""
    total = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            total += os.path.getsize(fp)
    return total / (1024 * 1024)


def copy_yolo_model(experiment_dir: str, models_dir: str) -> bool:
    """Copy best.pt to submission models directory."""
    src = os.path.join(experiment_dir, "weights", "best.pt")
    dst = os.path.join(models_dir, "helmet_detector.pt")

    if not os.path.exists(src):
        print(f"  ✗ best.pt not found at {src}")
        return False

    shutil.copy2(src, dst)
    size_mb = os.path.getsize(dst) / (1024 * 1024)
    print(f"  ✓ Copied best.pt → helmet_detector.pt ({size_mb:.1f} MB)")
    return True


def copy_paddleocr_models(models_dir: str) -> bool:
    """
    Copy PaddleOCR models from default cache to submission.
    PaddleOCR 3.x (PaddleX) caches models in ~/.paddlex/official_models/
    """
    paddlex_cache = os.path.expanduser("~/.paddlex/official_models")

    # PaddleOCR 3.x model names in ~/.paddlex/official_models/
    model_map = {
        "ocr_det":          "PP-OCRv5_server_det",
        "ocr_rec":          "en_PP-OCRv5_mobile_rec",
        "ocr_textline_ori": "PP-LCNet_x1_0_textline_ori",
    }

    ok = True
    for dst_name, src_name in model_map.items():
        src = os.path.join(paddlex_cache, src_name)
        dst = os.path.join(models_dir, dst_name)
        if os.path.isdir(src):
            if os.path.exists(dst):
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
            size = get_dir_size_mb(dst)
            print(f"  [OK] Copied PaddleOCR {dst_name} ({size:.1f} MB)")
        else:
            print(f"  [FAIL] PaddleOCR {dst_name} not found at {src}")
            print(f"         Run: python dev/scripts/download_ocr_models.py")
            ok = False

    return ok


def copy_easyocr_models(models_dir: str) -> bool:
    """Copy EasyOCR models from default cache to submission."""
    easyocr_cache = os.path.expanduser("~/.EasyOCR/model")
    dst = os.path.join(models_dir, "easyocr_models")
    os.makedirs(dst, exist_ok=True)

    required_files = ["english_g2.pth", "craft_mlt_25k.pth"]
    ok = True

    for fname in required_files:
        src = os.path.join(easyocr_cache, fname)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(dst, fname))
            size = os.path.getsize(os.path.join(dst, fname)) / (1024 * 1024)
            print(f"  ✓ Copied EasyOCR {fname} ({size:.1f} MB)")
        else:
            print(f"  ✗ EasyOCR {fname} not found at {src}")
            print(f"    Run: import easyocr; easyocr.Reader(['en']) to download")
            ok = False

    return ok


def check_solution_file(submission_dir: str) -> bool:
    """Validate solution.py has no forbidden patterns."""
    solution_path = os.path.join(submission_dir, "solution.py")
    ok = True

    if not os.path.exists(solution_path):
        print(f"  ✗ solution.py not found")
        return False

    with open(solution_path, "r") as f:
        content = f.read()

    # Check for custom package imports
    bad_imports = ["from src ", "from dev ", "from callbacks ", "from scripts "]
    for pattern in bad_imports:
        if pattern in content:
            print(f"  ✗ Forbidden import found: '{pattern.strip()}'")
            ok = False

    # Check for hardcoded paths
    bad_paths = ["/home/", "/Users/", "C:\\", "/mnt/"]
    for pattern in bad_paths:
        if pattern in content:
            print(f"  ✗ Hardcoded path found: '{pattern}'")
            ok = False

    # Check for network calls
    bad_network = ["requests.get", "urllib.request", "wget", "http://", "https://"]
    for pattern in bad_network:
        # Exclude comments and docstrings
        lines = content.split("\n")
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith("#") or stripped.startswith('"""') or stripped.startswith("'''"):
                continue
            if pattern in line and "download_enabled=False" not in line:
                print(f"  ✗ Network call found on line {i+1}: '{pattern}'")
                ok = False

    # Check class interface exists
    if "class TrafficViolationDetector" not in content:
        print(f"  ✗ TrafficViolationDetector class not found")
        ok = False

    if 'def predict(self' not in content:
        print(f"  ✗ predict() method not found")
        ok = False

    if 'def __init__(self' not in content:
        print(f"  ✗ __init__() method not found")
        ok = False

    if ok:
        print(f"  ✓ solution.py checks passed")

    return ok


def check_model_loading(submission_dir: str) -> bool:
    """Try to import and initialize the detector."""
    abs_submission = os.path.abspath(submission_dir)
    try:
        sys.path.insert(0, abs_submission)

        # Remove cached module if exists
        if "solution" in sys.modules:
            del sys.modules["solution"]

        from solution import TrafficViolationDetector

        models_dir = os.path.join(abs_submission, "models")
        print(f"  Loading from: {models_dir}")
        detector = TrafficViolationDetector(model_dir=models_dir)
        print(f"  [OK] Model initialization successful")
        return True
    except Exception as e:
        print(f"  [FAIL] Model initialization failed: {e}")
        return False
    finally:
        if abs_submission in sys.path:
            sys.path.remove(abs_submission)


def check_predict(submission_dir: str, test_image: str = None) -> bool:
    """Run a prediction to verify output format."""
    abs_submission = os.path.abspath(submission_dir)
    try:
        sys.path.insert(0, abs_submission)

        if "solution" in sys.modules:
            del sys.modules["solution"]

        from solution import TrafficViolationDetector

        models_dir = os.path.join(abs_submission, "models")
        detector = TrafficViolationDetector(model_dir=models_dir)

        # Use test image if provided, otherwise create a dummy in a temp dir
        if test_image and os.path.exists(test_image):
            img_path = test_image
        else:
            import tempfile
            import numpy as np
            import cv2
            dummy = np.zeros((640, 640, 3), dtype=np.uint8)
            tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
            img_path = tmp.name
            tmp.close()
            cv2.imwrite(img_path, dummy)

        start = time.time()
        output = detector.predict(img_path)
        elapsed = time.time() - start

        # Validate format
        assert isinstance(output, dict), f"Output must be dict, got {type(output)}"
        assert "violations" in output, "Missing 'violations' key"
        assert isinstance(output["violations"], list), "violations must be list"

        for v in output["violations"]:
            assert "num_riders" in v and isinstance(v["num_riders"], int), \
                "num_riders must be int"
            assert "helmet_violations" in v and isinstance(v["helmet_violations"], int), \
                "helmet_violations must be int"
            assert "license_plate" in v and isinstance(v["license_plate"], str), \
                "license_plate must be str"

        print(f"  ✓ predict() works ({elapsed:.2f}s)")
        print(f"    Output: {json.dumps(output, indent=2)}")
        return True

    except Exception as e:
        print(f"  ✗ predict() failed: {e}")
        return False
    finally:
        if abs_submission in sys.path:
            sys.path.remove(abs_submission)


def export(experiment_dir: str, submission_dir: str, roll_number: str = None,
           test_image: str = None):
    """Main export function."""

    if roll_number:
        submission_dir = os.path.join(os.path.dirname(submission_dir), roll_number)

    models_dir = os.path.join(submission_dir, "models")
    os.makedirs(models_dir, exist_ok=True)

    print("=" * 60)
    print("  EXPORT & PRE-SUBMISSION CHECKS")
    print("=" * 60)

    checks = {}

    # 1. Copy YOLO model
    print("\n1. Copying YOLO model...")
    checks["yolo_model"] = copy_yolo_model(experiment_dir, models_dir)

    # 2. Copy PaddleOCR models
    print("\n2. Copying PaddleOCR models...")
    checks["paddleocr"] = copy_paddleocr_models(models_dir)

    # 3. Copy EasyOCR models
    print("\n3. Copying EasyOCR models...")
    checks["easyocr"] = copy_easyocr_models(models_dir)

    # 4. Check model size
    print("\n4. Checking model size...")
    total_mb = get_dir_size_mb(models_dir)
    checks["model_size"] = total_mb <= 250
    status = "✓" if checks["model_size"] else "✗"
    print(f"  {status} Total models size: {total_mb:.1f} MB (limit: 250 MB)")

    # 5. Check solution.py
    print("\n5. Checking solution.py...")
    checks["solution_file"] = check_solution_file(submission_dir)

    # 6. Check required files
    print("\n6. Checking required files...")
    required = ["solution.py", "requirements.txt", "README.md"]
    for fname in required:
        exists = os.path.exists(os.path.join(submission_dir, fname))
        status = "✓" if exists else "✗"
        print(f"  {status} {fname}")
        checks[f"file_{fname}"] = exists

    # 7. Check model loading
    print("\n7. Testing model initialization...")
    if checks.get("yolo_model") and checks.get("paddleocr"):
        checks["model_loading"] = check_model_loading(submission_dir)
    else:
        print("  ⊘ Skipped (missing models)")
        checks["model_loading"] = False

    # 8. Check predict
    print("\n8. Testing predict()...")
    if checks.get("model_loading"):
        checks["predict"] = check_predict(submission_dir, test_image)
    else:
        print("  ⊘ Skipped (model loading failed)")
        checks["predict"] = False

    # Summary
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    all_pass = all(checks.values())
    for name, passed in checks.items():
        status = "✓" if passed else "✗"
        print(f"  {status} {name}")

    if all_pass:
        print(f"\n  ✅ ALL CHECKS PASSED — Submission ready!")
        print(f"  📦 Directory: {submission_dir}")
        print(f"  📦 Zip: cd {os.path.dirname(submission_dir)} && "
              f"zip -r {os.path.basename(submission_dir)}.zip {os.path.basename(submission_dir)}/")
    else:
        print(f"\n  ❌ SOME CHECKS FAILED — Fix before submitting")


def main():
    parser = argparse.ArgumentParser(description="Export model and validate submission")
    parser.add_argument("--experiment_dir", type=str, required=True,
                        help="Path to training experiment directory")
    parser.add_argument("--submission_dir", type=str, default="submission/ROLL_NUMBER",
                        help="Path to submission directory")
    parser.add_argument("--roll_number", type=str, default=None,
                        help="Roll number (creates subdirectory)")
    parser.add_argument("--test_image", type=str, default=None,
                        help="Path to test image for predict() check")
    args = parser.parse_args()
    export(args.experiment_dir, args.submission_dir, args.roll_number, args.test_image)


if __name__ == "__main__":
    main()
