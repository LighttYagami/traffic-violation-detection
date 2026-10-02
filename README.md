# Traffic Rule Violation Detection on Two-Wheelers

## Project Summary

This system detects traffic violations on two-wheelers (motorcycles/scooters) from street images. Given a single RGB image, it identifies vehicles where riders exceed two or are not wearing helmets, and reads their license plates.

The pipeline uses a single YOLO26s model to detect four object classes (bike, helmet, no-helmet, number-plate) simultaneously, then applies spatial association logic to group detections per vehicle and determine violations. License plates are read using PaddleOCR with EasyOCR as a fallback.

### Pipeline

```
Input Image
  → YOLO26s (4 classes, imgsz=1280 GPU / 640 CPU)
  → Detect: bike, helmet, no-helmet, number-plate
  → Associate detections to bikes using spatial priors
  → rider_count = helmet + no-helmet per bike
  → Violation if rider_count > 2 OR no-helmet > 0
  → For violating bikes: crop plate → preprocess → PaddleOCR / EasyOCR
  → Output JSON
```

### Output Format

```json
{
  "violations": [
    {
      "num_riders": 3,
      "helmet_violations": 1,
      "license_plate": "MH12AB1234"
    }
  ]
}
```

Non-violating vehicles are excluded. Images with no violations return `{"violations": []}`.

---

## Setup

### Option A: Automatic (recommended)

Run the download script to fetch all model weights:

```bash
pip install -r requirements.txt
python download_models.py
```

This downloads and extracts all models into the `models/` directory automatically.

### Option B: Manual

1. Download the models archive from: https://huggingface.co/rajcreo/traffic-violation-models/tree/main

2. Extract into the `models/` directory so the structure looks like:

```
models/
├── helmet_detector.pt
├── paddleocr/
│   ├── PP-OCRv5_server_det/
│   │   ├── config.json
│   │   ├── inference.json
│   │   ├── inference.pdiparams
│   │   └── inference.yml
│   ├── en_PP-OCRv5_mobile_rec/
│   │   ├── config.json
│   │   ├── inference.json
│   │   ├── inference.pdiparams
│   │   └── inference.yml
│   └── PP-LCNet_x1_0_textline_ori/
│       ├── config.json
│       ├── inference.json
│       ├── inference.pdiparams
│       └── inference.yml
└── easyocr_models/
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Usage

```python
from solution import TrafficViolationDetector

model = TrafficViolationDetector(model_dir="./models")
output = model.predict("path/to/image.jpg")
print(output)
```

---

## Model Details

| Component | Model | Size | Purpose |
|-----------|-------|------|---------|
| Detection | YOLO26s (fine-tuned) | ~51 MB | Detect bike, helmet, no-helmet, number-plate |
| OCR (primary) | PaddleOCR v5 | ~97 MB | License plate text recognition |
| OCR (fallback) | EasyOCR | ~30 MB | Fallback when PaddleOCR confidence < 0.3 |
| **Total** | | **~178 MB** | **Within 250 MB limit** |

### Detection

- Single-model architecture detecting all 4 classes simultaneously
- Trained on 5,216 images (Roboflow Bike Helmet Detection dataset)
- imgsz=1280 (GPU) / 640 (CPU) for adaptive inference
- mAP@50: 0.943, per-class AP@50: bike 0.923, helmet 0.955, no-helmet 0.930, plate 0.964

### Association Logic

- Helmets/no-helmets assigned to bikes if their center falls within the upper 70% of the bike bbox
- Plates assigned if center falls within lower 60% of bike bbox
- Overlapping bikes resolved by nearest-center assignment
- Rider count capped at 5

### OCR Strategy

- PaddleOCR runs first (~50-100ms per crop)
- EasyOCR runs only if PaddleOCR confidence < 0.3
- Rotation retry at ±5° and ±10° if both engines produce low confidence
- Always returns best available text (low-confidence output beats empty string for edit distance)

---

## Dependencies

- Python 3.8+
- PyTorch
- ultralytics (YOLO26)
- paddleocr / paddlepaddle
- easyocr
- opencv-python
- numpy

See `requirements.txt` for exact versions.

---

## File Structure

```
<ROLL_NUMBER>/
├── solution.py           # Complete pipeline (single file)
├── models/               # Model weights (≤250 MB)
├── download_models.py    # Auto-download script for models
├── requirements.txt      # Python dependencies
└── README.md             # This file
```

---

## Hardware Adaptation

The system automatically adapts to available hardware:
- **GPU available**: Uses CUDA, imgsz=1280 for higher accuracy on small objects
- **CPU only**: Falls back to imgsz=640 for reasonable inference time

---

## Error Handling

- `predict()` never raises exceptions; returns `{"violations": []}` on any failure
- Corrupt/missing images are handled gracefully
- OCR failures return empty string (not an exception)
- All models are loaded offline with no network calls
