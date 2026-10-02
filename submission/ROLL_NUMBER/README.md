# Traffic Rule Violation Detection — Two-Wheelers

## Approach

Single-model pipeline using YOLO26s for detecting traffic violations on two-wheelers.

### Pipeline

1. **Detection**: YOLO26s detects 4 classes — `bike`, `helmet`, `no-helmet`, `number-plate` — in a single forward pass at 1280px resolution (GPU) or 640px (CPU).

2. **Association**: For each detected bike bounding box, helmet/no-helmet/plate detections are associated using spatial containment (center-point-inside-bbox) with geometric priors:
   - Helmet/no-helmet must be in upper 70% of bike bbox
   - Plate must be in lower 60% of bike bbox
   - Overlapping bikes resolved by nearest-center assignment

3. **Violation Check**: A violation exists if `num_riders > 2` OR `helmet_violations > 0`. Rider count = helmet detections + no-helmet detections. Non-violating bikes are excluded from output.

4. **OCR**: For violating bikes only, the license plate crop is preprocessed (resize, CLAHE, Gaussian blur) and read using dual OCR — PaddleOCR first, EasyOCR as fallback if confidence < 0.3.

### Models

| Model | File | Size | Purpose |
|-------|------|------|---------|
| YOLO26s | helmet_detector.pt | ~20 MB | Object detection (4 classes) |
| PaddleOCR | ocr_det/, ocr_rec/, ocr_cls/ | ~12 MB | Primary plate text recognition |
| EasyOCR | easyocr_models/ | ~12 MB | Fallback plate text recognition |

Total model size: ~55-60 MB

### Classes

```
0: bike
1: helmet
2: no-helmet
3: number-plate
```

### Dataset

Fine-tuned on "Bike Helmet Detection" dataset from Roboflow (~6,500 images after subsampling). Full parameter update, 50 epochs, RTX 6000.

### Known Limitations

- Rider count relies on visible helmet/no-helmet detections; fully occluded riders may be missed
- OCR accuracy degrades on very small, blurry, or severely angled plates
- Dense overlapping traffic scenes may cause association errors
