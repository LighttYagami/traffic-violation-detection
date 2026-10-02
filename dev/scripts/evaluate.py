"""
evaluate.py
=============
End-to-end pipeline evaluation with deep accuracy analysis.
Runs the full pipeline (YOLO → association → OCR) on test images,
computes pipeline-level metrics, and generates failure case visualizations.

Different from test_solution.py:
- test_solution.py: validates format + timing (pre-submission sanity check)
- evaluate.py: deep accuracy analysis with ground truth, failure cases, annotated images

Usage:
    # Evaluate using submission solution.py
    python evaluate.py \
        --submission_dir ../../submission/ROLL_NUMBER \
        --test_images_dir ../data/test/images \
        --ground_truth ../data/test/ground_truth.json \
        --output_dir ../evaluation_results

    # Evaluate directly with a trained YOLO model (YOLO-only, no OCR)
    python evaluate.py \
        --model_weights ../training_logs/exp_001/weights/best.pt \
        --test_images_dir ../data/test/images \
        --test_labels_dir ../data/test/labels \
        --output_dir ../evaluation_results

Ground truth JSON format (for full pipeline evaluation):
{
    "image_name.jpg": {
        "violations": [
            {
                "num_riders": 3,
                "helmet_violations": 1,
                "license_plate": "MH12AB1234"
            }
        ]
    }
}
"""

import os
import sys
import json
import time
import argparse
import cv2
import numpy as np
from glob import glob
from collections import defaultdict

# ─── Visualization constants ──────────────────────────────────────────────────
CLASS_NAMES = ["bike", "helmet", "no-helmet", "number-plate"]
GT_COLOR   = (0, 200, 0)    # green  — ground truth boxes
PRED_COLOR = (0, 80, 255)   # red    — predicted boxes


# ─── Metrics ─────────────────────────────────────────────────────────────────


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


def normalized_edit_distance_score(pred: str, gt: str) -> float:
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


def compute_pipeline_metrics(predictions: dict, ground_truth: dict) -> dict:
    """
    Compute all pipeline-level evaluation metrics.

    Returns dict with:
    - violation_detection_rate
    - false_positive_rate
    - rider_count_accuracy
    - rider_count_mae
    - helmet_count_accuracy
    - helmet_count_mae
    - ocr_exact_match
    - ocr_edit_distance
    - final_score (w1=0.5, w2=0.5)
    - per_image_results (detailed per-image breakdown)
    """
    total_gt_violations = 0
    total_pred_violations = 0
    detected_violations = 0
    false_positives = 0

    correct_rider_count = 0
    rider_count_errors = []

    correct_helmet_count = 0
    helmet_count_errors = []

    ocr_exact_matches = 0
    ocr_edit_scores = []

    per_image_results = {}

    # Images in ground truth
    for img_name, gt in ground_truth.items():
        pred = predictions.get(img_name, {"violations": []})

        gt_violations = gt.get("violations", [])
        pred_violations = pred.get("violations", [])

        total_gt_violations += len(gt_violations)
        total_pred_violations += len(pred_violations)

        img_result = {
            "gt_count": len(gt_violations),
            "pred_count": len(pred_violations),
            "matched": [],
            "missed_gt": [],
            "false_pos": [],
            "failure_types": [],
        }

        # Match GT to predictions (ordered matching)
        matched_pred_indices = set()
        for i, gt_v in enumerate(gt_violations):
            if i < len(pred_violations):
                pred_v = pred_violations[i]
                matched_pred_indices.add(i)
                detected_violations += 1

                match_detail = {
                    "gt": gt_v,
                    "pred": pred_v,
                    "rider_correct": False,
                    "helmet_correct": False,
                    "ocr_score": 0.0,
                }

                # Rider count
                gt_riders = gt_v["num_riders"]
                pred_riders = pred_v["num_riders"]
                if pred_riders == gt_riders:
                    correct_rider_count += 1
                    match_detail["rider_correct"] = True
                else:
                    img_result["failure_types"].append("wrong_rider_count")
                rider_count_errors.append(abs(pred_riders - gt_riders))

                # Helmet violations
                gt_helmets = gt_v["helmet_violations"]
                pred_helmets = pred_v["helmet_violations"]
                if pred_helmets == gt_helmets:
                    correct_helmet_count += 1
                    match_detail["helmet_correct"] = True
                else:
                    img_result["failure_types"].append("wrong_helmet_count")
                helmet_count_errors.append(abs(pred_helmets - gt_helmets))

                # OCR
                gt_plate = gt_v.get("license_plate", "").strip().upper()
                pred_plate = pred_v.get("license_plate", "").strip().upper()

                if gt_plate == pred_plate:
                    ocr_exact_matches += 1

                edit_score = normalized_edit_distance_score(pred_plate, gt_plate)
                ocr_edit_scores.append(edit_score)
                match_detail["ocr_score"] = edit_score

                if edit_score < 0.7 and gt_plate:
                    img_result["failure_types"].append("ocr_failure")

                img_result["matched"].append(match_detail)
            else:
                img_result["missed_gt"].append(gt_v)
                img_result["failure_types"].append("missed_violation")

        # False positives (predicted violations not matched to GT)
        for j, pred_v in enumerate(pred_violations):
            if j not in matched_pred_indices:
                false_positives += 1
                img_result["false_pos"].append(pred_v)
                img_result["failure_types"].append("false_positive")

        # Deduplicate failure types
        img_result["failure_types"] = list(set(img_result["failure_types"]))
        per_image_results[img_name] = img_result

    # Images in predictions but not ground truth (extra false positives)
    for img_name, pred in predictions.items():
        if img_name not in ground_truth:
            pred_violations = pred.get("violations", [])
            if pred_violations:
                false_positives += len(pred_violations)
                per_image_results[img_name] = {
                    "gt_count": 0,
                    "pred_count": len(pred_violations),
                    "matched": [],
                    "missed_gt": [],
                    "false_pos": pred_violations,
                    "failure_types": ["false_positive"],
                }

    total = max(total_gt_violations, 1)
    metrics = {
        "total_gt_violations": total_gt_violations,
        "total_pred_violations": total_pred_violations,
        "detected_violations": detected_violations,
        "false_positives": false_positives,
        "violation_detection_rate": detected_violations / total,
        "false_positive_rate": false_positives / max(total_pred_violations, 1),
        "rider_count_accuracy": correct_rider_count / total,
        "rider_count_mae": np.mean(rider_count_errors) if rider_count_errors else 0.0,
        "helmet_count_accuracy": correct_helmet_count / total,
        "helmet_count_mae": np.mean(helmet_count_errors) if helmet_count_errors else 0.0,
        "ocr_exact_match": ocr_exact_matches / total,
        "ocr_edit_distance_avg": float(np.mean(ocr_edit_scores)) if ocr_edit_scores else 0.0,
        "per_image_results": per_image_results,
    }

    # Combined score (project spec)
    w1, w2 = 0.5, 0.5
    violation_score = (metrics["rider_count_accuracy"] + metrics["helmet_count_accuracy"]) / 2
    metrics["violation_score"] = violation_score
    metrics["ocr_score"] = metrics["ocr_edit_distance_avg"]
    metrics["final_score"] = w1 * violation_score + w2 * metrics["ocr_edit_distance_avg"]

    return metrics


# ─── Failure Case Analysis ───────────────────────────────────────────────────


def categorize_failures(per_image_results: dict) -> dict:
    """
    Categorize failure cases for analysis.
    Returns dict: {category: [list of (img_name, details)]}
    """
    categories = {
        "missed_violation": [],
        "false_positive": [],
        "wrong_rider_count": [],
        "wrong_helmet_count": [],
        "ocr_failure": [],
    }

    for img_name, result in per_image_results.items():
        for failure_type in result.get("failure_types", []):
            if failure_type in categories:
                categories[failure_type].append((img_name, result))

    return categories


def load_gt_boxes(label_path: str, img_w: int, img_h: int,
                  class_names: list = CLASS_NAMES) -> list:
    """
    Read a YOLO-format label file and return pixel bounding boxes.
    Returns list of (class_name, x1, y1, x2, y2).
    """
    boxes = []
    if not os.path.exists(label_path):
        return boxes
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls_id = int(parts[0])
            cx, cy, bw, bh = map(float, parts[1:5])
            x1 = int((cx - bw / 2) * img_w)
            y1 = int((cy - bh / 2) * img_h)
            x2 = int((cx + bw / 2) * img_w)
            y2 = int((cy + bh / 2) * img_h)
            name = class_names[cls_id] if cls_id < len(class_names) else str(cls_id)
            boxes.append((name, x1, y1, x2, y2))
    return boxes


def get_pred_boxes(yolo_model, img_path: str, conf: float = 0.25) -> list:
    """
    Run YOLO inference and return detected bounding boxes.
    Returns list of (class_name, confidence, x1, y1, x2, y2).
    """
    results = yolo_model(img_path, conf=conf, verbose=False)
    boxes = []
    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls.item())
            conf_val = float(box.conf.item())
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            name = yolo_model.names.get(cls_id, str(cls_id))
            boxes.append((name, conf_val, x1, y1, x2, y2))
    return boxes


def _draw_box_with_label(img, label: str, x1: int, y1: int, x2: int, y2: int,
                         color: tuple, font_scale: float = 0.42) -> None:
    """Draw a bounding box with a filled label background (in-place)."""
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX,
                                         font_scale, 1)
    ly = max(y1 - th - baseline - 4, 0)
    cv2.rectangle(img, (x1, ly), (x1 + tw + 4, ly + th + baseline + 4), color, -1)
    cv2.putText(img, label, (x1 + 2, ly + th + 2),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1, cv2.LINE_AA)


def _put_text_with_bg(img, text: str, x: int, y: int, color: tuple,
                      font_scale: float = 0.48) -> int:
    """Draw text with a semi-transparent dark background. Returns new y offset."""
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
    overlay = img.copy()
    cv2.rectangle(overlay, (x - 2, y - th - 4), (x + tw + 4, y + 4), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, img, 0.45, 0, img)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, color, 1, cv2.LINE_AA)
    return y + th + 8


def draw_annotations(image, gt_violations: list, pred_violations: list,
                     gt_boxes: list = None, pred_boxes: list = None):
    """
    Draw bounding boxes and violation-level summary text on image.

    gt_boxes  — list of (class_name, x1, y1, x2, y2)         from label file
    pred_boxes — list of (class_name, conf, x1, y1, x2, y2)  from YOLO inference
    """
    img = image.copy()

    # ── Bounding boxes ────────────────────────────────────────────────────────
    if gt_boxes:
        for (name, x1, y1, x2, y2) in gt_boxes:
            _draw_box_with_label(img, f"GT: {name}", x1, y1, x2, y2, GT_COLOR)

    if pred_boxes:
        for (name, conf, x1, y1, x2, y2) in pred_boxes:
            _draw_box_with_label(img, f"{name} {conf:.2f}", x1, y1, x2, y2, PRED_COLOR)

    # ── Violation-level text summary (top-left) ───────────────────────────────
    y = 20
    for i, v in enumerate(gt_violations):
        plate = v.get("license_plate", "") or ""
        text = (f"GT  bike{i}: riders={v['num_riders']}  "
                f"viol={v['helmet_violations']}  plate={plate}")
        y = _put_text_with_bg(img, text, 8, y, GT_COLOR)

    y += 4
    for i, v in enumerate(pred_violations):
        plate = v.get("license_plate", "") or ""
        text = (f"Pred bike{i}: riders={v['num_riders']}  "
                f"viol={v['helmet_violations']}  plate={plate}")
        y = _put_text_with_bg(img, text, 8, y, PRED_COLOR)

    return img


def save_failure_cases(categories: dict, test_images_dir: str,
                       ground_truth: dict, predictions: dict,
                       output_dir: str, max_per_category: int = 20,
                       test_labels_dir: str = None, yolo_model=None):
    """
    Save annotated failure case images organized by failure type.

    Overlays on each image:
      - Green boxes: ground truth detections (from YOLO label files if test_labels_dir given)
      - Red boxes:   YOLO predicted detections (if yolo_model given)
      - Text summary: violation-level GT vs prediction comparison

    Output structure:
    output_dir/failure_cases/
    ├── missed_violation/
    │   ├── img_001.jpg
    │   └── img_001.json
    ├── false_positive/
    ├── wrong_rider_count/
    ├── wrong_helmet_count/
    └── ocr_failure/
    """
    failure_dir = os.path.join(output_dir, "failure_cases")

    for category, cases in categories.items():
        if not cases:
            continue

        cat_dir = os.path.join(failure_dir, category)
        os.makedirs(cat_dir, exist_ok=True)

        for img_name, result in cases[:max_per_category]:
            img_path = os.path.join(test_images_dir, img_name)
            if not os.path.exists(img_path):
                continue

            image = cv2.imread(img_path)
            if image is None:
                continue

            img_h, img_w = image.shape[:2]

            # Ground-truth bounding boxes from label file
            gt_boxes = None
            if test_labels_dir:
                stem = os.path.splitext(img_name)[0]
                lbl_path = os.path.join(test_labels_dir, stem + ".txt")
                gt_boxes = load_gt_boxes(lbl_path, img_w, img_h)

            # Predicted bounding boxes from YOLO
            pred_boxes = None
            if yolo_model is not None:
                pred_boxes = get_pred_boxes(yolo_model, img_path)

            gt = ground_truth.get(img_name, {"violations": []})
            pred = predictions.get(img_name, {"violations": []})

            annotated = draw_annotations(
                image,
                gt.get("violations", []),
                pred.get("violations", []),
                gt_boxes=gt_boxes,
                pred_boxes=pred_boxes,
            )

            save_path = os.path.join(cat_dir, img_name)
            cv2.imwrite(save_path, annotated)

            json_path = os.path.join(cat_dir, os.path.splitext(img_name)[0] + ".json")
            detail = {
                "image": img_name,
                "failure_type": category,
                "ground_truth": gt,
                "prediction": pred,
                "analysis": result,
            }
            with open(json_path, "w") as f:
                json.dump(detail, f, indent=2, default=str)

        print(f"  {category}: saved {min(len(cases), max_per_category)} failure cases")


# ─── YOLO-Only Evaluation ────────────────────────────────────────────────────


def evaluate_yolo_only(model_weights: str, test_images_dir: str,
                       test_labels_dir: str, output_dir: str):
    """
    Evaluate YOLO model directly on test set (no OCR, no association).
    Uses Ultralytics val() for standard detection metrics.
    """
    from ultralytics import YOLO
    import yaml

    print("=" * 60)
    print("  YOLO-ONLY EVALUATION")
    print("=" * 60)

    model = YOLO(model_weights)

    # Create a temporary data.yaml pointing to test data
    temp_yaml = os.path.join(output_dir, "temp_eval_data.yaml")
    data_config = {
        "path": os.path.dirname(test_images_dir),
        "train": "images",  # dummy, not used
        "val": "images",
        "test": "images",
        "nc": 4,
        "names": ["bike", "helmet", "no-helmet", "number-plate"],
    }

    # Check if test_labels_dir is alongside test_images_dir
    parent = os.path.dirname(test_images_dir)
    if os.path.basename(test_images_dir) == "images":
        data_config["path"] = parent

    os.makedirs(output_dir, exist_ok=True)
    with open(temp_yaml, "w") as f:
        yaml.dump(data_config, f, default_flow_style=False)

    # Run validation
    metrics = model.val(
        data=temp_yaml,
        split="val",
        plots=True,
        conf=0.25,
        iou=0.5,
        verbose=True,
        project=output_dir,
        name="yolo_eval",
        exist_ok=True,
    )

    # Save results
    results = {
        "model": model_weights,
        "mAP50": float(metrics.box.map50),
        "mAP50_95": float(metrics.box.map),
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
    }

    if hasattr(model, "names"):
        results["per_class"] = {}
        for i, name in model.names.items():
            try:
                results["per_class"][name] = {
                    "ap50": float(metrics.box.ap50[i]),
                    "ap50_95": float(metrics.box.maps[i]),
                }
            except (IndexError, AttributeError):
                pass

    report_path = os.path.join(output_dir, "yolo_eval_report.json")
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n  mAP@50:    {results['mAP50']:.4f}")
    print(f"  mAP@50-95: {results['mAP50_95']:.4f}")
    print(f"  Precision: {results['precision']:.4f}")
    print(f"  Recall:    {results['recall']:.4f}")

    if "per_class" in results:
        print(f"\n  Per-class AP@50:")
        for name, vals in results["per_class"].items():
            print(f"    {name:15s}: {vals['ap50']:.4f}")

    print(f"\n  Report saved: {report_path}")

    # Clean up temp yaml
    os.remove(temp_yaml)

    return results


# ─── Full Pipeline Evaluation ────────────────────────────────────────────────


def evaluate_pipeline(submission_dir: str, test_images_dir: str,
                      ground_truth_path: str, output_dir: str,
                      test_labels_dir: str = None):
    """
    Full pipeline evaluation: detection → association → OCR → metrics.

    test_labels_dir — optional path to YOLO label files (.txt) for the test
                      images; when provided, failure-case images will show
                      ground-truth bounding boxes in green.
    """
    print("=" * 60)
    print("  FULL PIPELINE EVALUATION")
    print("=" * 60)

    # Find test images
    test_images = sorted(glob(os.path.join(test_images_dir, "*.jpg")))
    if not test_images:
        test_images = sorted(glob(os.path.join(test_images_dir, "*.png")))
    print(f"  Test images: {len(test_images)}")

    # Load ground truth
    with open(ground_truth_path) as f:
        ground_truth = json.load(f)
    print(f"  Ground truth entries: {len(ground_truth)}")

    # Import and initialize detector
    sys.path.insert(0, submission_dir)
    if "solution" in sys.modules:
        del sys.modules["solution"]
    from solution import TrafficViolationDetector

    models_dir = os.path.join(submission_dir, "models")
    print(f"\n  Initializing detector...")
    init_start = time.time()
    detector = TrafficViolationDetector(model_dir=models_dir)
    init_time = time.time() - init_start
    print(f"  Init time: {init_time:.2f}s")

    # Run predictions
    print(f"\n  Running predictions...")
    predictions = {}
    timings = []

    for i, img_path in enumerate(test_images):
        img_name = os.path.basename(img_path)

        start = time.time()
        output = detector.predict(img_path)
        elapsed = time.time() - start

        timings.append(elapsed)
        predictions[img_name] = output

        n_viol = len(output.get("violations", []))
        if (i + 1) % 50 == 0 or n_viol > 0:
            print(f"    [{i+1}/{len(test_images)}] {img_name}: "
                  f"{n_viol} violations ({elapsed:.2f}s)")

    # Save predictions
    os.makedirs(output_dir, exist_ok=True)
    pred_path = os.path.join(output_dir, "predictions.json")
    with open(pred_path, "w") as f:
        json.dump(predictions, f, indent=2)

    # ── Timing Report ──
    print(f"\n{'='*60}")
    print(f"  TIMING REPORT")
    print(f"{'='*60}")
    print(f"  Mean:    {np.mean(timings):.3f}s")
    print(f"  Median:  {np.median(timings):.3f}s")
    print(f"  Max:     {np.max(timings):.3f}s")
    print(f"  P95:     {np.percentile(timings, 95):.3f}s")
    print(f"  Over 5s: {sum(1 for t in timings if t > 5.0)}/{len(timings)}")

    # ── Pipeline Metrics ──
    print(f"\n{'='*60}")
    print(f"  PIPELINE METRICS")
    print(f"{'='*60}")

    metrics = compute_pipeline_metrics(predictions, ground_truth)

    print(f"\n  Violation Detection:")
    print(f"    GT violations:         {metrics['total_gt_violations']}")
    print(f"    Predicted violations:  {metrics['total_pred_violations']}")
    print(f"    Correctly detected:    {metrics['detected_violations']}")
    print(f"    False positives:       {metrics['false_positives']}")
    print(f"    Detection rate:        {metrics['violation_detection_rate']:.4f}")
    print(f"    False positive rate:   {metrics['false_positive_rate']:.4f}")

    print(f"\n  Rider Count:")
    print(f"    Exact match accuracy:  {metrics['rider_count_accuracy']:.4f}")
    print(f"    Mean absolute error:   {metrics['rider_count_mae']:.4f}")

    print(f"\n  Helmet Violations:")
    print(f"    Exact match accuracy:  {metrics['helmet_count_accuracy']:.4f}")
    print(f"    Mean absolute error:   {metrics['helmet_count_mae']:.4f}")

    print(f"\n  OCR:")
    print(f"    Exact match rate:      {metrics['ocr_exact_match']:.4f}")
    print(f"    Avg edit distance:     {metrics['ocr_edit_distance_avg']:.4f}")

    print(f"\n  ─────────────────────────────────")
    print(f"  Violation score (w1):    {metrics['violation_score']:.4f}")
    print(f"  OCR score (w2):          {metrics['ocr_score']:.4f}")
    print(f"  FINAL SCORE:             {metrics['final_score']:.4f}")

    # ── Failure Case Analysis ──
    print(f"\n{'='*60}")
    print(f"  FAILURE CASE ANALYSIS")
    print(f"{'='*60}")

    per_image = metrics.pop("per_image_results")
    categories = categorize_failures(per_image)

    print(f"\n  Failure distribution:")
    for cat, cases in categories.items():
        print(f"    {cat:25s}: {len(cases)} images")

    # Auto-derive labels dir from images dir if not given (sibling folder)
    if test_labels_dir is None:
        parent = os.path.dirname(test_images_dir)
        candidate = os.path.join(parent, "labels")
        if os.path.isdir(candidate):
            test_labels_dir = candidate
            print(f"  Auto-detected labels dir: {test_labels_dir}")

    # Load YOLO model for predicted bbox visualization in failure images
    yolo_model = None
    yolo_weights = os.path.join(submission_dir, "models", "helmet_detector.pt")
    if os.path.exists(yolo_weights):
        try:
            from ultralytics import YOLO
            yolo_model = YOLO(yolo_weights)
            print(f"  Loaded YOLO model for bbox visualization: {yolo_weights}")
        except Exception as e:
            print(f"  [WARN] Could not load YOLO for visualization: {e}")

    # Save failure case images
    print(f"\n  Saving failure case images...")
    save_failure_cases(
        categories, test_images_dir, ground_truth, predictions, output_dir,
        test_labels_dir=test_labels_dir,
        yolo_model=yolo_model,
    )

    # ── Failure Summary ──
    # Most common failure patterns
    failure_combos = defaultdict(int)
    for img_name, result in per_image.items():
        types = tuple(sorted(result.get("failure_types", [])))
        if types:
            failure_combos[types] += 1

    if failure_combos:
        print(f"\n  Most common failure patterns:")
        for combo, count in sorted(failure_combos.items(), key=lambda x: -x[1])[:10]:
            print(f"    {' + '.join(combo):45s}: {count} images")

    # ── Save Full Report ──
    # Remove per_image from metrics for cleaner report (it's large)
    report = {
        "metrics": metrics,
        "timing": {
            "mean_seconds": float(np.mean(timings)),
            "median_seconds": float(np.median(timings)),
            "max_seconds": float(np.max(timings)),
            "p95_seconds": float(np.percentile(timings, 95)),
            "over_5s_count": int(sum(1 for t in timings if t > 5.0)),
            "init_time_seconds": init_time,
        },
        "failure_counts": {cat: len(cases) for cat, cases in categories.items()},
        "failure_patterns": {
            " + ".join(k): v
            for k, v in sorted(failure_combos.items(), key=lambda x: -x[1])[:20]
        },
    }

    report_path = os.path.join(output_dir, "eval_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    # Save per-image details separately (large file)
    detail_path = os.path.join(output_dir, "per_image_results.json")
    with open(detail_path, "w") as f:
        json.dump(per_image, f, indent=2, default=str)

    print(f"\n  Reports saved:")
    print(f"    {report_path}")
    print(f"    {detail_path}")
    print(f"    {pred_path}")
    print(f"    {os.path.join(output_dir, 'failure_cases/')}")

    print(f"\n{'='*60}")
    if metrics["final_score"] >= 0.7:
        print(f"  ✅ FINAL SCORE: {metrics['final_score']:.4f} — Looking good!")
    elif metrics["final_score"] >= 0.4:
        print(f"  ⚠️  FINAL SCORE: {metrics['final_score']:.4f} — Needs improvement")
    else:
        print(f"  ❌ FINAL SCORE: {metrics['final_score']:.4f} — Significant issues")
    print("=" * 60)

    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate traffic violation detection pipeline"
    )
    parser.add_argument("--submission_dir", type=str, 
                        default=r"C:\Users\rajcr\Desktop\Study\MTech\Courses\CV\Project\source_code\traffic_violation_detection\submission\ROLL_NUMBER",
                        help="Path to submission directory (for full pipeline eval)")
    parser.add_argument("--model_weights", type=str, default=None,
                        help="Path to YOLO weights (for YOLO-only eval)")
    parser.add_argument("--test_images_dir", type=str, 
                        default=r"C:\Users\rajcr\Desktop\Study\MTech\Courses\CV\Project\Dataset\GroundTruthData\images",
                        help="Path to test images directory")
    parser.add_argument("--test_labels_dir", type=str, default=None,
                        help="Path to test labels directory (for YOLO-only eval)")
    parser.add_argument("--ground_truth", type=str, 
                        default=r"C:\Users\rajcr\Desktop\Study\MTech\Courses\CV\Project\Dataset\GroundTruthData\ground_truth.json",
                        help="Path to ground truth JSON (for full pipeline eval)")
    parser.add_argument("--output_dir", type=str, 
                        default=r"C:\Users\rajcr\Desktop\Study\MTech\Courses\CV\Project\source_code\traffic_violation_detection\dev\evaluation_results",
                        help="Path to output directory")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.model_weights:
        # YOLO-only evaluation
        evaluate_yolo_only(
            model_weights=args.model_weights,
            test_images_dir=args.test_images_dir,
            test_labels_dir=args.test_labels_dir,
            output_dir=args.output_dir,
        )

    if args.submission_dir and args.ground_truth:
        # Full pipeline evaluation
        evaluate_pipeline(
            submission_dir=args.submission_dir,
            test_images_dir=args.test_images_dir,
            ground_truth_path=args.ground_truth,
            output_dir=args.output_dir,
            test_labels_dir=args.test_labels_dir,
        )

    if not args.model_weights and not (args.submission_dir and args.ground_truth):
        print("Provide either:")
        print("  --model_weights (for YOLO-only eval)")
        print("  --submission_dir + --ground_truth (for full pipeline eval)")
        print("  Both (for both evaluations)")


if __name__ == "__main__":
    main()
