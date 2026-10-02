# Traffic Violation Detection — Project Plan
# =================================================

---

## 1. DESIGN OVERVIEW — THE FINAL PIPELINE

```
Input Image
    │
    ▼
┌───────────────────────────────────────────┐
│  YOLO26s (single model, imgsz=1280)       │
│  Detects: bike, helmet, no-helmet, plate  │
│  All 4 classes in one pass                │
└───────────────────────────────────────────┘
    │
    ▼
┌───────────────────────────────────────────┐
│  Step 1: Find all bike detections         │
│  No bikes? → return {"violations": []}    │
└───────────────────────────────────────────┘
    │
    ▼
┌───────────────────────────────────────────┐
│  Step 2: For each bike bbox, find which   │
│  helmet / no-helmet / plate detections    │
│  fall inside it (center-point check)      │
│                                           │
│  rider_count = helmet + no-helmet         │
│  helmet_violations = no-helmet count      │
│                                           │
│  Geometric priors:                        │
│  - helmet/no-helmet in upper portion      │
│  - plate in lower portion                 │
│  - rider count capped at 5 (sanity)       │
│  - plate-to-bike: pick nearest if         │
│    multiple bikes overlap                 │
└───────────────────────────────────────────┘
    │
    ▼
┌───────────────────────────────────────────┐
│  Step 3: Check violations                 │
│  Violation = rider_count > 2              │
│          OR helmet_violations > 0         │
│                                           │
│  No violations? → return {"violations":[]}│
└───────────────────────────────────────────┘
    │ (only for violating bikes)
    ▼
┌───────────────────────────────────────────┐
│  Step 4: For each violating bike,         │
│  crop the associated plate region         │
│  - Generous padding (15-20%)              │
│  - Preprocess: resize, CLAHE,             │
│    grayscale, mild Gaussian blur          │
│                                           │
│  Run dual OCR (conditional):              │
│  - PaddleOCR first                        │
│  - EasyOCR only if PaddleOCR conf < 0.3   │
│  - If both low: retry with ±5° rotation   │
│  - Return best available text             │
│    (even low-conf beats empty string      │
│     for edit distance scoring)            │
└───────────────────────────────────────────┘
    │
    ▼
  Output JSON: {"violations": [...]}
```

### WHY SINGLE MODEL, NOT TWO-STAGE

We considered a two-stage approach (Stage 1: detect bikes+riders,
Stage 2: crop and detect helmet/no-helmet/plate on crops). Rejected because:

1. Cascading failure — if Stage 1 misses a bike, everything downstream is zero
2. Crop quality is fragile — too tight clips helmets, too loose bleeds neighbors
3. Coordinate remapping errors in label preprocessing
4. Double training work, double debugging surface
5. Double inference time (4 YOLO passes for 3 bikes vs 1 pass)
6. YOLO26's built-in STAL (Small-Target-Aware Label Assignment) already
   addresses the small-object problem that two-stage was meant to solve
7. imgsz=1280 gives 4x pixel area over 640, further reducing small-object issues
8. Dataset is already annotated for single-model (all 4 classes in same labels)

---

## 2. MODEL CHOICES

### Detection: YOLO26s

Why YOLO26 over alternatives:

| Factor              | YOLOv11s       | YOLOv12s           | YOLO26s              |
|---------------------|----------------|--------------------|----------------------|
| Accuracy (COCO mAP) | Good           | +1.2% over v11     | Comparable to v12    |
| Training stability  | Stable         | Unstable (known)   | Stable (MuSGD)       |
| Training speed      | Fast           | 20% slower         | Fast                 |
| CPU inference       | Baseline       | Slower             | 43% faster           |
| Small objects       | Standard       | Standard           | Built-in STAL         |
| NMS dependency      | Yes            | Yes                | NMS-free              |
| Ecosystem           | Very mature    | Fragmented repos   | Ultralytics-native    |
| Roboflow support    | Full           | Full               | Full (YOLO26 format)  |

YOLO26s key advantages for this project:
- STAL specifically designed for small object detection (helmets, plates)
- 43% faster CPU inference — handles unknown evaluator hardware
- NMS-free — fewer moving parts, less latency, simpler deployment
- MuSGD optimizer — stable training, faster convergence
- Same Ultralytics API: `from ultralytics import YOLO; YOLO("yolo26s.pt")`
- Model size: ~20 MB, well within 250 MB budget

### OCR: Dual Engine (Conditional)

- PaddleOCR (primary): better on clean, well-lit plates, multi-language
- EasyOCR (fallback): runs ONLY when PaddleOCR confidence < 0.3
- Conditional approach saves inference time on CPU
- Both pre-downloaded for offline evaluation
- Combined model size: ~35-40 MB

### Total Model Budget

```
models/
├── helmet_detector.pt          # ~20 MB  — Fine-tuned YOLO26s
├── ocr_det/                    # ~3-4 MB — PaddleOCR detection
├── ocr_rec/                    # ~5-8 MB — PaddleOCR recognition
├── ocr_cls/                    # ~2-3 MB — PaddleOCR angle classifier
└── easyocr_models/             # ~10-15 MB — EasyOCR english model
    ├── english_g2.pth
    └── craft_mlt_25k.pth

Total: ~55-60 MB (well within 250 MB limit)
```

---

## 3. DATASET

### Source

Single primary dataset: "Bike Helmet Detection" from Roboflow
- Total: 8,466 images
- Classes: bike, helmet, no-helmet, number-plate
- All 4 classes annotated in the same label files
- Pre-labeled in YOLO format

### Dataset Composition & Why Everything Stays

| Image Type                        | Count     | Keep? | Reasoning |
|-----------------------------------|-----------|-------|-----------|
| Real street view (top/side/back)  | ~3,500    | Yes (subsample) | Core training data, but subsample to ~1,500-2,000 to prevent viewpoint overfitting |
| Single bike front view            | 500       | Yes   | Diverse angles and real street scenes |
| Person face closeup               | 500       | Yes   | Teaches no-helmet head features; no false positives because pipeline starts with bike detection — faces without bikes are never processed |
| Dashcam / GoPro view              | 700       | Yes   | Valuable viewpoint diversity |
| Bicycle with bicycle helmet       | 300       | Yes   | Only labeled as helmet/no-helmet (no "bike" class label), so helps helmet learning without causing bicycle false positives in bike detection |
| Street photo person walking       | 50        | Yes   | Teaches no-helmet features; only 50 images, negligible impact |
| Video calling photo               | Rare      | Yes   | Only ~10-20 images, not worth filtering effort |
| Night view street image           | Very few  | Yes   | Critical for robustness despite low count |

### What Changes

Only one intervention: **subsample the 3,500 same-camera frames** (naming
prefix "frame_") to ~1,500-2,000 images. These are sequential video frames
from one camera position. Without subsampling, 40%+ of training data shares
one viewpoint, causing the model to overfit to that perspective.

### Train / Val / Test Split

- **Test set**: Use the existing test images in the dataset (already diverse —
  real street, night, bicycle, walking, mixed everything). No data leakage
  confirmed — no overlap with the 3,500 same-camera training frames.
- **Train / Val**: Split remaining images 85% / 15% using stratified splitting.
  Stratify by violation type (number of helmets, no-helmets per image) so
  each split has similar class proportions.
- **Same-camera grouping**: All frames from one video sequence go entirely
  into train or val, never split across both (prevents data leakage).

### Rider Count Strategy

Rider count = helmet detections + no-helmet detections within each bike bbox.

No separate "rider" or "person" class needed because:
- Dataset has no rider annotations (would need manual annotation or new dataset)
- helmet + no-helmet covers the vast majority of cases
- Edge case (occluded rider head) is rare and hard even for dedicated person detectors
- If rider counting accuracy is poor after testing, fallback: use COCO
  pretrained person detection from YOLO26s on bike crops (zero extra training)

---

## 4. TRAINING

### Hardware

- GPU: RTX 6000, 48 GB VRAM
- Batch size: 64 (can push to 128)
- Expected training time: ~25-35 min for 5k images, 50 epochs

### Fine-tuning Strategy

Full parameter update (not LoRA). Reasons:
- YOLO26s has only ~9-10M parameters — tiny model, no need for parameter-efficient methods
- LoRA is designed for Transformers (Q/K/V projections), not CNN-dominant architectures
- Dataset is large enough (~6,000+ images after subsampling) to avoid overfitting
- Full fine-tuning lets backbone learn domain-specific features (Indian traffic,
  helmet shapes, plate styles)

### Training Configuration

```yaml
experiment_name: "exp_001_yolo26s_full_finetune"

model:
  architecture: "yolo26s.pt"       # YOLO26s pretrained on COCO
  input_size: 1280                 # Higher res for small objects (helmets, plates)

training:
  epochs: 50
  batch_size: 64
  patience: 10                     # Early stopping
  amp: true                        # Mixed precision
  freeze_layers: 0                 # Full fine-tuning
  optimizer: auto                  # YOLO26 uses MuSGD by default
  learning_rate:
    initial: 0.01
    final_factor: 0.01
    warmup_epochs: 3

augmentation:
  hsv_h: 0.015                    # Hue shift
  hsv_s: 0.7                      # Saturation shift
  hsv_v: 0.4                      # Brightness shift (helps low-light robustness)
  fliplr: 0.5                     # Horizontal flip
  mosaic: 1.0                     # Mosaic augmentation
  flipud: 0.0                     # No vertical flip (bikes don't flip)
  degrees: 0.0                    # No rotation
  # Motion blur: add via albumentations if time permits

checkpointing:
  save_period: 10                  # Save checkpoint every 10 epochs
  save_best: true                  # Always save best.pt by mAP
  save_last: true                  # Always save last.pt for resume
```

### Training Logging — What Gets Captured

**Automatic (Ultralytics saves results.csv):**
- Per epoch: train/box_loss, train/cls_loss, train/dfl_loss
- Per epoch: val/box_loss, val/cls_loss, val/dfl_loss
- Per epoch: precision, recall, mAP@50, mAP@50-95
- Per epoch: learning rate (pg0, pg1, pg2)

**Custom callback (saves custom_metrics.json):**
- Gradient L2 norms grouped by backbone / neck / head
- Gradient max absolute values
- Weight L2 norms per group
- Per-class AP (bike, helmet, no-helmet, plate) at each epoch
- Wall-clock time per epoch
- All flushed to disk atomically after every epoch (crash-safe)

### Checkpoint Strategy

| Trigger                     | What Saved          | Where                              |
|-----------------------------|--------------------|------------------------------------|
| Every epoch                 | last.pt (overwrite) | weights/last.pt                    |
| Best mAP epoch              | best.pt (overwrite) | weights/best.pt                    |
| Every 10 epochs             | epoch_N.pt          | weights/epoch_checkpoints/         |
| Training end                | both best + last    | weights/                           |

All checkpoints include optimizer state for resumable training.

---

## 5. ASSOCIATION LOGIC — DETAILED DESIGN

### Core Algorithm

```
For each bike bbox detected:
    1. Find all helmet detections whose CENTER falls inside bike bbox
    2. Find all no-helmet detections whose CENTER falls inside bike bbox
    3. Find the license plate detection with highest overlap with bike bbox

    rider_count = len(helmets) + len(no_helmets)
    helmet_violations = len(no_helmets)

    Apply geometric priors:
    - Discard helmet/no-helmet if not in upper 60% of bike bbox
    - Discard plate if not in lower 50% of bike bbox
    - Cap rider_count at 5 (sanity — no real bike has 6+ riders)

    If rider_count > 2 OR helmet_violations > 0:
        → This bike is a violation
        → Process plate for OCR
```

### Handling Overlapping Bikes

When two bike bboxes overlap (common in traffic):
- A helmet/no-helmet detection that falls inside BOTH bike bboxes →
  assign to the bike whose CENTER is closer
- A plate detection inside both → assign to the bike whose BOTTOM EDGE
  is closer (plates are at the rear/bottom of bikes)
- This prevents double-counting riders across overlapping bikes

### Edge Cases

| Scenario                          | Behavior                               |
|-----------------------------------|----------------------------------------|
| Bike detected, zero helmets/no-helmets | Not a violation (empty bike / parked)  |
| No bikes detected in image        | Return {"violations": []}              |
| Violation found but no plate      | Return license_plate: ""               |
| OCR completely fails              | Return license_plate: ""               |
| Image fails to load               | Return {"violations": []}              |

---

## 6. OCR PIPELINE — DETAILED DESIGN

### Plate Preprocessing

```
1. Crop plate region from image with 15-20% padding on all sides
2. Resize to standard height (64px), maintain aspect ratio
3. Convert to grayscale
4. Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
5. Apply mild Gaussian blur (kernel=3) — reduces compression artifacts
```

### Dual OCR — Conditional Execution

```
plate_crop → preprocess
    │
    ├──→ PaddleOCR → (text_A, conf_A)
    │
    │    If conf_A > 0.3 → return cleaned text_A ✓ DONE
    │
    ├──→ conf_A ≤ 0.3 → run EasyOCR → (text_B, conf_B)
    │
    │    If conf_B > 0.3 → return cleaned text_B ✓ DONE
    │
    ├──→ Both low → retry with rotated crops (±5°)
    │    Run PaddleOCR on each rotation
    │    If any conf > 0.5 → return that text ✓ DONE
    │
    └──→ All attempts low → return best available text
         (even low-conf text beats "" for edit distance scoring)
```

### Why Conditional (Not Always Both)

- On GPU: both engines total ~100-200ms, negligible
- On CPU: both engines total ~800-1200ms per plate
- Conditional saves ~500-800ms per plate on CPU
- Most plates are readable by PaddleOCR alone (>70% above 0.3 conf)
- EasyOCR only runs on hard cases — the ~30% where it actually helps

### Adaptive Image Size for Device

```python
if torch.cuda.is_available():
    self.imgsz = 1280    # Full resolution for GPU
else:
    self.imgsz = 640     # Reduced for CPU to stay under 5s
```

### Text Cleaning

```
- Strip whitespace and newlines
- Remove special characters except alphanumeric, hyphen, space
- Common OCR error fixes: O→0, I→1, B→8, S→5, Z→2
- Apply fixes conservatively — only in positions where a digit is expected
```

---

## 7. CODEBASE STRUCTURE

### Two Separate Zones

```
traffic_violation_detection/          ← DEVELOPMENT (your machine only)
│
├── dev/
│   ├── config/
│   │   └── train_config.yaml         # All hyperparameters
│   │
│   ├── data/
│   │   ├── raw/                      # Downloaded dataset
│   │   ├──                 # After subsampling same-camera frames
│   │   │   ├── train/images/, labels/
│   │   │   ├── val/images/, labels/
│   │   │   └── test/images/, labels/
│   │   └── data.yaml                 # YOLO dataset config
│   │
│   ├── scripts/
│   │   ├── download_data.py          # Downloads dataset from Roboflow
│   │   ├── subsample_frames.py       # Subsamples 3,500 same-camera frames
│   │   ├── split_dataset.py          # Stratified train/val split
│   │   ├── verify_dataset.py         # Sanity checks: corrupts, class balance
│   │   ├── train.py                  # Main training script with callbacks
│   │   ├── evaluate.py               # End-to-end pipeline test
│   │   └── export_model.py           # Copies best.pt → submission, runs checks
│   │
│   ├── callbacks/
│   │   └── training_logger.py        # Logs gradients, weight norms, per-class AP
│   │
│   ├── notebooks/
│   │   ├── 01_data_exploration.ipynb       # Class distribution, sample images
│   │   ├── 02_training_analysis.ipynb      # Loss curves, gradients, LR, convergence
│   │   ├── 03_evaluation_analysis.ipynb    # Confusion matrix, PR curves, failures
│   │   ├── 04_experiment_comparison.ipynb  # Side-by-side training runs
│   │   └── 05_ocr_analysis.ipynb           # PaddleOCR vs EasyOCR comparison
│   │
│   ├── training_logs/
│   │   └── exp_XXX/
│   │       ├── weights/
│   │       │   ├── best.pt
│   │       │   ├── last.pt
│   │       │   └── epoch_checkpoints/
│   │       ├── metrics/
│   │       │   ├── results.csv             # Ultralytics auto-saved
│   │       │   ├── custom_metrics.json     # Our callback
│   │       │   └── per_class_ap.json
│   │       ├── plots/
│   │       ├── config_snapshot.yaml
│   │       └── train.log
│   │
│   └── evaluation_results/
│       ├── predictions/
│       ├── visualizations/
│       ├── eval_report.json
│       └── failure_cases/
│
├── submission/                       ← SUBMISSION (exactly what gets submitted)
│   └── <ROLL_NUMBER>/
│       ├── solution.py               # SINGLE FILE — entire pipeline
│       ├── models/                   # ≤ 250 MB total
│       │   ├── helmet_detector.pt
│       │   ├── ocr_det/
│       │   ├── ocr_rec/
│       │   ├── ocr_cls/
│       │   └── easyocr_models/
│       ├── requirements.txt
│       └── README.md
│
└── tests/
    ├── test_solution.py              # Simulates exact evaluator behavior
    └── test_images/
```

### Why This Separation

The evaluator runs ONLY:
```python
from solution import TrafficViolationDetector
model = TrafficViolationDetector(model_dir="./models")
output = model.predict("some_image.jpg")
```

No dev/, no callbacks/, no notebooks/. Just solution.py and models/.
Everything else is for your development and report.

---

## 8. solution.py — SELF-CONTAINED SINGLE FILE

```
solution.py structure:
│
├── Imports (standard library + pip packages only)
│   import os, json, cv2, numpy, torch
│   from ultralytics import YOLO
│   from paddleocr import PaddleOCR
│   import easyocr
│
├── Helper Functions
│   ├── _load_image(path)
│   ├── _get_detections(yolo_results)
│   ├── _compute_iou(box1, box2)
│   ├── _center_inside_box(inner_center, outer_box)
│   ├── _associate_detections_to_bikes(bikes, helmets, no_helmets, plates)
│   │     For each bike bbox:
│   │       Find helmet/no-helmet with center inside bike bbox
│   │       Apply geometric priors (upper/lower region checks)
│   │       Handle overlapping bikes (assign to nearest)
│   │       Find associated plate
│   ├── _is_violation(rider_count, helmet_violations)
│   ├── _crop_plate(image, plate_bbox, padding=0.15)
│   ├── _preprocess_plate(crop)
│   │     Resize → grayscale → CLAHE → Gaussian blur
│   ├── _rotate_crop(crop, angle)
│   ├── _run_paddle_ocr(engine, crop)
│   ├── _run_easy_ocr(engine, crop)
│   ├── _run_dual_ocr(paddle, easy, crop)
│   │     PaddleOCR first → if low conf → EasyOCR → if both low → rotation retry
│   ├── _clean_plate_text(raw_text)
│   └── _build_output(violations)
│
├── TrafficViolationDetector Class
│   ├── __init__(self, model_dir="./models")
│   │     Load YOLO26s from model_dir
│   │     Load PaddleOCR from model_dir (offline, show_log=False)
│   │     Load EasyOCR from model_dir (download_enabled=False)
│   │     Set adaptive imgsz (1280 GPU / 640 CPU)
│   │     Set thresholds (conf=0.25, iou=0.5, overlap=0.3)
│   │     Map class IDs (must match data.yaml from training)
│   │
│   └── predict(self, image_path) → dict
│         try:
│           1. Load image
│           2. Run YOLO26s → get all detections
│           3. Separate by class: bikes, helmets, no_helmets, plates
│           4. Associate detections to bikes
│           5. Check violations for each bike
│           6. For violating bikes: crop plate → preprocess → dual OCR
│           7. Return {"violations": [...]}
│         except Exception:
│           return {"violations": []}
```

### Critical Rules

1. ZERO imports from your own packages
2. ALL paths through model_dir parameter
3. predict() NEVER raises exceptions
4. predict() is stateless — no caching between calls
5. No network calls — all models pre-downloaded
6. No print statements (or gated behind verbose=False)
7. Only violating bikes reported — non-violations excluded

---

## 9. EVALUATION — THREE LEVELS

### Level 1: YOLO Validation (automatic during training)

Runs after every epoch. Computes mAP@50, mAP@50-95, precision, recall.
Used for early stopping and best.pt selection. No code needed.

### Level 2: Test Set Evaluation (after training)

```python
model = YOLO("training_logs/exp_XXX/weights/best.pt")
model.val(data="data.yaml", split="test", plots=True, conf=0.25)
```

Generates: confusion matrix, PR curves, F1 curves, per-class metrics.

### Level 3: Full Pipeline Test (simulates evaluator)

```python
# Import from submission directory exactly as evaluator would
from solution import TrafficViolationDetector
detector = TrafficViolationDetector(model_dir="submission/<ROLL>/models")

for image_path in test_images:
    output = detector.predict(image_path)
    # Validate output format
    # Measure timing (must be < 5s)
    # Compare with ground truth if available
```

Checks: output format correctness, inference timing, accuracy metrics,
failure case analysis with annotated images.

### CPU Benchmark (before submission)

```bash
CUDA_VISIBLE_DEVICES="" python test_solution.py
```

Run 20 images with GPU disabled. Verify all under 5 seconds.
If any exceed 4s, the adaptive imgsz (640 on CPU) handles it.

---

## 10. ANALYSIS NOTEBOOKS

### dev/notebooks/01_data_exploration.ipynb
- Class distribution bar chart
- Bbox size distribution (are plates tiny?)
- Sample grid with GT annotations
- Image source composition (same-camera vs dashcam vs others)
- Co-occurrence analysis

### dev/notebooks/02_training_analysis.ipynb
Reads from: results.csv + custom_metrics.json (NO retraining)
- Loss curves (box/cls/dfl, train vs val)
- mAP convergence with best epoch marked
- Per-class AP over epochs (bike/helmet/no-helmet/plate)
- Learning rate schedule
- Gradient norms (backbone/neck/head)
- Weight norm evolution
- Epoch timing

### dev/notebooks/03_evaluation_analysis.ipynb
- Confusion matrix heatmap
- Per-class precision/recall
- PR curves overlaid
- Confidence threshold sensitivity (mAP vs conf)
- Failure case gallery (GT green, pred red)
- Detection size vs confidence scatter
- End-to-end timing breakdown

### dev/notebooks/04_experiment_comparison.ipynb
- Summary table across experiments
- Overlay mAP curves
- Per-class AP comparison
- Gradient norm comparison

### dev/notebooks/05_ocr_analysis.ipynb
- PaddleOCR vs EasyOCR per-plate comparison
- Dual vs single engine accuracy table
- Disagreement gallery
- Failure mode analysis (blurry, small, angled)
- Preprocessing impact (with/without CLAHE)

### Plot Style (shared across notebooks)
```python
COLORS = {
    'bike': '#2196F3',          # blue
    'helmet': '#4CAF50',        # green
    'no-helmet': '#F44336',     # red
    'number-plate': '#FF9800',  # orange
}
FIGSIZE = (10, 6)
SAVE_DPI = 150
# All plots saved as PNG + PDF
```

---

## 11. TRAINING CALLBACK — WHAT GETS LOGGED

### custom_metrics.json structure after training

```json
{
  "experiment": "exp_001_yolo26s_full_finetune",
  "model": "yolo26s.pt",
  "dataset": "Bike Helmet Detection (Roboflow)",
  "total_epochs_run": 38,
  "early_stopped": true,
  "best_epoch": 28,
  "training_time_seconds": 1680,

  "train_losses": [
    {"epoch": 0, "box_loss": 1.234, "cls_loss": 3.456, "dfl_loss": 1.678}
  ],

  "val_metrics": [
    {"epoch": 0, "precision": 0.423, "recall": 0.312, "mAP50": 0.356, "mAP50_95": 0.198}
  ],

  "per_class_ap": [
    {
      "epoch": 0,
      "bike":         {"ap50": 0.45, "precision": 0.52, "recall": 0.38},
      "helmet":       {"ap50": 0.38, "precision": 0.44, "recall": 0.31},
      "no-helmet":    {"ap50": 0.29, "precision": 0.35, "recall": 0.24},
      "number-plate": {"ap50": 0.31, "precision": 0.37, "recall": 0.27}
    }
  ],

  "learning_rates": [
    {"epoch": 0, "pg0": 0.00033, "pg1": 0.00033, "pg2": 0.00033}
  ],

  "gradients": [
    {
      "epoch": 0,
      "backbone": {"l2_norm": 0.156, "max_abs": 0.423, "mean_abs": 0.008},
      "neck":     {"l2_norm": 0.234, "max_abs": 0.567, "mean_abs": 0.015},
      "head":     {"l2_norm": 0.389, "max_abs": 0.891, "mean_abs": 0.028}
    }
  ],

  "weight_norms": [
    {"epoch": 0, "backbone_l2": 42.56, "neck_l2": 11.23, "head_l2": 7.89}
  ],

  "epoch_durations_seconds": [45.2, 43.8, 44.1]
}
```

### How the callback hooks into training

Gradient logging happens in on_train_epoch_end — after loss.backward()
computes gradients but BEFORE optimizer.step() consumes them. This is
the correct moment to read param.grad values.

Parameter grouping for YOLO26s:
- model.model[0] to model.model[9]   → backbone
- model.model[10] to model.model[21] → neck
- model.model[22] onwards            → head

All writes use atomic rename (write to .tmp, then os.replace) so a
crash mid-write never corrupts the JSON file.

---

## 12. REQUIREMENTS.TXT

```
ultralytics>=8.4.0
paddlepaddle>=2.5.0
paddleocr>=2.7.0
easyocr>=1.7.0
opencv-python>=4.8.0
numpy>=1.24.0
torch>=2.0.0
torchvision>=0.15.0
```

---

## 13. PRE-SUBMISSION CHECKLIST

```
MANDATORY — verify ALL before submitting:

□ submission/<ROLL_NUMBER>/ has exactly:
    solution.py, models/, requirements.txt, README.md

□ solution.py has NO imports from custom packages
    Verify: grep -n "from src\|from dev\|from callbacks" solution.py

□ models/ total ≤ 250 MB
    Verify: du -sh submission/<ROLL_NUMBER>/models/

□ All model paths use model_dir parameter
    Verify: grep -n "models/" solution.py (only os.path.join)

□ No internet calls
    Verify: grep -rn "download\|http\|requests\|urllib" solution.py

□ predict() never raises exceptions
    Verify: run on 100+ images, count exceptions (must be 0)

□ Output format matches spec exactly
    Verify: jsonschema validation on every output

□ Only violations reported (non-violating bikes excluded)
    Verify: test on clean image → {"violations": []}

□ Inference < 5s per image (GPU and CPU)
    Verify: timing test on 50+ images both modes

□ PaddleOCR models pre-downloaded (no auto-download)
    Verify: CUDA_VISIBLE_DEVICES="" + no internet → still works

□ EasyOCR models pre-downloaded (download_enabled=False)
    Verify: same offline test

□ Class IDs in solution.py match data.yaml from training
    Verify: print model.names after loading

□ requirements.txt tested in clean virtualenv
    Verify: fresh venv → pip install → import → predict()

□ No hardcoded absolute paths
    Verify: grep -rn "/home\|/Users\|C:\\" solution.py

□ README.md documents approach, models used, class names
```

---

## 14. EXECUTION TIMELINE

### Day 1: Dataset + First Training Run
- Download dataset from Roboflow (YOLO26 format)
- Subsample same-camera frames (3,500 → ~1,500)
- Stratified train/val split
- Verify dataset (class distribution, sample check)
- First training run: YOLO26s, 50 epochs, imgsz=1280, batch=64
- Expected time: ~30-40 min on RTX 6000

### Day 2: Pipeline Integration
- Build solution.py (single file, all logic)
- Integrate PaddleOCR + EasyOCR (pre-download models)
- Implement association logic + geometric priors
- Test end-to-end pipeline on sample images

### Day 3: Evaluation + Tuning
- Run Level 3 evaluation (full pipeline test)
- Tune confidence thresholds
- Analyze failure cases
- Second training run if needed (different hyperparams)
- CPU benchmark test

### Day 4: Polish + Submit
- Run notebooks for report figures
- Finalize README.md
- Export model to submission directory
- Run full pre-submission checklist
- Package and submit

---

## 15. KEY DESIGN DECISIONS — SUMMARY

| Decision              | Choice                      | Alternatives Considered     | Why This Won                               |
|-----------------------|-----------------------------|-----------------------------|-------------------------------------------|
| Detection model       | YOLO26s                     | YOLOv11s, YOLOv12s          | STAL for small objects, 43% faster CPU, NMS-free, stable training |
| Architecture          | Single model, 4 classes     | Two-stage (detect → crop → detect) | Simpler, no cascading failures, dataset already structured for it |
| Input resolution      | 1280 (GPU) / 640 (CPU)     | Fixed 640 or fixed 1280     | Adaptive handles unknown evaluator hardware |
| Fine-tuning           | Full parameter update       | LoRA, freeze backbone       | Small model (9-10M params), enough data, LoRA doesn't apply to CNNs |
| OCR strategy          | PaddleOCR primary, EasyOCR fallback | Single engine, always both | Conditional saves CPU time, covers edge cases |
| Rider counting        | helmet + no-helmet count    | Dedicated rider detection   | No rider annotations in dataset, practical baseline |
| Dataset filtering     | Keep almost everything      | Heavy filtering             | Pipeline's bike-first logic prevents false positives from non-bike images |
| Same-camera frames    | Subsample 3,500 → ~1,500   | Keep all, remove all        | Prevents viewpoint overfitting while keeping real traffic data |
| Test split            | Use existing test images    | Own random split            | Already diverse, no data leakage confirmed |
| Association           | Center-point-in-bbox + priors | IoU-based, graph matching  | Simple, fast, geometric priors handle 90%+ of cases |
| Plate preprocessing   | CLAHE + Gaussian blur + padding | Super-resolution, deblurring | Practical, fast, no extra model weight |
| Solution structure    | Single solution.py          | Multi-file with src/        | Evaluator only sees solution.py — multi-file imports would crash |
| Visualization         | Jupyter notebooks           | Python scripts              | Interactive, inline plots, doubles as report material |
```
