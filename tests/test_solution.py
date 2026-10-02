"""
test_solution.py
=================
Simulates the exact evaluation protocol from the project spec.
Tests on GPU and optionally on CPU. Validates output format and timing.

Usage:
    # Test with GPU
    python test_solution.py \
        --submission_dir ../submission/ROLL_NUMBER \
        --test_images_dir ./test_images

    # Test with CPU only
    CUDA_VISIBLE_DEVICES="" python test_solution.py \
        --submission_dir ../submission/ROLL_NUMBER \
        --test_images_dir ./test_images

    # Test with ground truth
    python test_solution.py \
        --submission_dir ../submission/ROLL_NUMBER \
        --test_images_dir ./test_images \
        --ground_truth ground_truth.json
"""

import os
import sys
import json
import time
import argparse
import numpy as np
from glob import glob


def levenshtein_distance(s1: str, s2: str) -> int:
    """Compute Levenshtein edit distance between two strings."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)

    prev_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        curr_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = prev_row[j + 1] + 1
            deletions = curr_row[j] + 1
            substitutions = prev_row[j] + (c1 != c2)
            curr_row.append(min(insertions, deletions, substitutions))
        prev_row = curr_row

    return prev_row[-1]


def normalized_edit_distance(pred: str, gt: str) -> float:
    """
    Normalized edit distance score.
    1.0 = exact match, 0.0 = completely wrong.
    """
    if not pred and not gt:
        return 1.0
    max_len = max(len(pred), len(gt))
    if max_len == 0:
        return 1.0
    dist = levenshtein_distance(pred, gt)
    return 1.0 - dist / max_len


def validate_output(output: dict, image_name: str) -> list:
    """Validate output format. Returns list of error strings."""
    errors = []

    if not isinstance(output, dict):
        errors.append(f"{image_name}: output is not a dict")
        return errors

    if "violations" not in output:
        errors.append(f"{image_name}: missing 'violations' key")
        return errors

    if not isinstance(output["violations"], list):
        errors.append(f"{image_name}: 'violations' is not a list")
        return errors

    for i, v in enumerate(output["violations"]):
        prefix = f"{image_name} violation[{i}]"

        if not isinstance(v, dict):
            errors.append(f"{prefix}: not a dict")
            continue

        if "num_riders" not in v:
            errors.append(f"{prefix}: missing 'num_riders'")
        elif not isinstance(v["num_riders"], int):
            errors.append(f"{prefix}: 'num_riders' is not int")

        if "helmet_violations" not in v:
            errors.append(f"{prefix}: missing 'helmet_violations'")
        elif not isinstance(v["helmet_violations"], int):
            errors.append(f"{prefix}: 'helmet_violations' is not int")

        if "license_plate" not in v:
            errors.append(f"{prefix}: missing 'license_plate'")
        elif not isinstance(v["license_plate"], str):
            errors.append(f"{prefix}: 'license_plate' is not str")

    return errors


def compute_metrics(predictions: dict, ground_truth: dict) -> dict:
    """
    Compute pipeline-level evaluation metrics.
    
    Metrics:
    - violation_detection_rate: fraction of GT violations detected
    - rider_count_accuracy: exact match rate for num_riders
    - helmet_count_accuracy: exact match rate for helmet_violations
    - ocr_exact_match: exact string match rate
    - ocr_edit_distance: mean normalized edit distance
    """
    total_gt_violations = 0
    detected_violations = 0
    correct_rider_count = 0
    correct_helmet_count = 0
    ocr_exact_matches = 0
    ocr_edit_scores = []

    for img_name, gt in ground_truth.items():
        pred = predictions.get(img_name, {"violations": []})

        gt_violations = gt.get("violations", [])
        pred_violations = pred.get("violations", [])

        total_gt_violations += len(gt_violations)

        # Simple ordered matching (GT[i] matches pred[i])
        for i, gt_v in enumerate(gt_violations):
            if i < len(pred_violations):
                detected_violations += 1
                pred_v = pred_violations[i]

                # Rider count
                if pred_v["num_riders"] == gt_v["num_riders"]:
                    correct_rider_count += 1

                # Helmet violations
                if pred_v["helmet_violations"] == gt_v["helmet_violations"]:
                    correct_helmet_count += 1

                # OCR
                gt_plate = gt_v.get("license_plate", "").strip().upper()
                pred_plate = pred_v.get("license_plate", "").strip().upper()

                if gt_plate == pred_plate:
                    ocr_exact_matches += 1

                edit_score = normalized_edit_distance(pred_plate, gt_plate)
                ocr_edit_scores.append(edit_score)

    total = max(total_gt_violations, 1)
    metrics = {
        "total_gt_violations": total_gt_violations,
        "detected_violations": detected_violations,
        "violation_detection_rate": detected_violations / total,
        "rider_count_accuracy": correct_rider_count / total,
        "helmet_count_accuracy": correct_helmet_count / total,
        "ocr_exact_match": ocr_exact_matches / total,
        "ocr_edit_distance": np.mean(ocr_edit_scores) if ocr_edit_scores else 0.0,
    }

    # Combined score (as per project spec)
    w1, w2 = 0.5, 0.5
    violation_score = (metrics["rider_count_accuracy"] + metrics["helmet_count_accuracy"]) / 2
    metrics["final_score"] = w1 * violation_score + w2 * metrics["ocr_edit_distance"]

    return metrics


def run_test(submission_dir: str, test_images_dir: str,
             ground_truth_path: str = None):
    """Main test function."""

    # ── Find test images ──
    image_extensions = ("*.jpg", "*.jpeg", "*.png")
    test_images = []
    for ext in image_extensions:
        test_images.extend(glob(os.path.join(test_images_dir, ext)))
    test_images = sorted(test_images)

    if not test_images:
        print(f"No test images found in {test_images_dir}")
        return

    print("=" * 60)
    print("  PIPELINE EVALUATION")
    print("=" * 60)
    print(f"  Submission: {submission_dir}")
    print(f"  Test images: {len(test_images)}")
    print(f"  Device: {'GPU' if 'CUDA_VISIBLE_DEVICES' not in os.environ or os.environ.get('CUDA_VISIBLE_DEVICES') != '' else 'CPU'}")

    # ── Import and initialize (exactly as evaluator does) ──
    sys.path.insert(0, submission_dir)
    if "solution" in sys.modules:
        del sys.modules["solution"]

    from solution import TrafficViolationDetector

    models_dir = os.path.join(submission_dir, "models")
    print(f"\n  Initializing model...")
    init_start = time.time()
    detector = TrafficViolationDetector(model_dir=models_dir)
    init_time = time.time() - init_start
    print(f"  Init time: {init_time:.2f}s")

    # ── Run predictions ──
    print(f"\n  Running predictions...")
    predictions = {}
    timings = []
    format_errors = []

    for img_path in test_images:
        img_name = os.path.basename(img_path)

        start = time.time()
        output = detector.predict(img_path)
        elapsed = time.time() - start

        timings.append(elapsed)
        predictions[img_name] = output

        # Validate format
        errors = validate_output(output, img_name)
        format_errors.extend(errors)

        # Progress
        n_violations = len(output.get("violations", []))
        status = f"{n_violations} violations" if n_violations > 0 else "clean"
        overtime = " ⚠ SLOW" if elapsed > 5.0 else ""
        print(f"    {img_name}: {status} ({elapsed:.2f}s){overtime}")

    # ── Timing report ──
    print(f"\n{'='*60}")
    print(f"  TIMING REPORT")
    print(f"{'='*60}")
    print(f"  Mean:    {np.mean(timings):.3f}s")
    print(f"  Median:  {np.median(timings):.3f}s")
    print(f"  Max:     {np.max(timings):.3f}s")
    print(f"  P95:     {np.percentile(timings, 95):.3f}s")
    print(f"  Over 5s: {sum(1 for t in timings if t > 5.0)}/{len(timings)}")

    # ── Format validation ──
    print(f"\n{'='*60}")
    print(f"  FORMAT VALIDATION")
    print(f"{'='*60}")
    if format_errors:
        print(f"  ✗ {len(format_errors)} format errors:")
        for err in format_errors[:10]:
            print(f"    - {err}")
    else:
        print(f"  ✓ All outputs have correct format")

    # ── Accuracy metrics (if ground truth available) ──
    if ground_truth_path and os.path.exists(ground_truth_path):
        print(f"\n{'='*60}")
        print(f"  ACCURACY METRICS")
        print(f"{'='*60}")

        with open(ground_truth_path) as f:
            ground_truth = json.load(f)

        metrics = compute_metrics(predictions, ground_truth)

        print(f"  Violation detection rate:  {metrics['violation_detection_rate']:.4f}")
        print(f"  Rider count accuracy:      {metrics['rider_count_accuracy']:.4f}")
        print(f"  Helmet count accuracy:     {metrics['helmet_count_accuracy']:.4f}")
        print(f"  OCR exact match:           {metrics['ocr_exact_match']:.4f}")
        print(f"  OCR edit distance (avg):   {metrics['ocr_edit_distance']:.4f}")
        print(f"  ─────────────────────────")
        print(f"  FINAL SCORE:               {metrics['final_score']:.4f}")

        # Save report
        report_path = os.path.join(test_images_dir, "eval_report.json")
        with open(report_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"\n  Report saved: {report_path}")
    else:
        print(f"\n  No ground truth provided. Skipping accuracy metrics.")
        print(f"  To compute accuracy, create a ground_truth.json and pass --ground_truth")

    # ── Save predictions ──
    pred_path = os.path.join(test_images_dir, "predictions.json")
    with open(pred_path, "w") as f:
        json.dump(predictions, f, indent=2)
    print(f"  Predictions saved: {pred_path}")

    # ── Final verdict ──
    print(f"\n{'='*60}")
    all_under_5s = all(t < 5.0 for t in timings)
    no_format_errors = len(format_errors) == 0

    if all_under_5s and no_format_errors:
        print(f"  ✅ PASS — All timing and format checks passed")
    else:
        if not all_under_5s:
            print(f"  ❌ TIMING FAIL — {sum(1 for t in timings if t >= 5.0)} images over 5s")
        if not no_format_errors:
            print(f"  ❌ FORMAT FAIL — {len(format_errors)} format errors")
    print("=" * 60)


def main():

    parser = argparse.ArgumentParser(description="Test submission pipeline")
    parser.add_argument("--submission_dir", 
                        type=str, 
                        default=r"C:\Users\rajcr\Desktop\Study\MTech\Courses\CV\Project\submission\MT2025721")
    parser.add_argument("--test_images_dir", 
                        type=str, 
                        default=r"C:\Users\rajcr\Desktop\Study\MTech\Courses\CV\Project\Dataset\GroundTruthData\images")
    parser.add_argument("--ground_truth", type=str, default=None)
    args = parser.parse_args()
    run_test(args.submission_dir, args.test_images_dir, args.ground_truth)


if __name__ == "__main__":
    main()
