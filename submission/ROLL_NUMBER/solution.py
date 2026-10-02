"""
Traffic Violation Detection - Solution
=======================================
Single self-contained file for detecting traffic violations on two-wheelers.

Pipeline:
    1. YOLO26s detects: bike, helmet, no-helmet, number-plate
    2. For each bike bbox, find helmet/no-helmet/plate detections inside it
    3. Check violations: num_riders > 2 OR helmet_violations > 0
    4. For violating bikes, crop plate region and run dual OCR

Models required in model_dir:
    - helmet_detector.pt       (fine-tuned YOLO)
    - ocr_det/                 (PaddleOCR PP-OCRv5_server_det)
    - ocr_rec/                 (PaddleOCR en_PP-OCRv5_mobile_rec)
    - ocr_textline_ori/        (PaddleOCR PP-LCNet_x1_0_textline_ori)
    - easyocr_models/          (EasyOCR english model weights)
"""

import os

# Must be set before any paddle import to avoid OneDNN PIR errors on Windows
os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["FLAGS_enable_pir_api"] = "0"

import cv2
import numpy as np
import torch
from ultralytics import YOLO


# ─── Lazy OCR imports (only loaded when needed) ──────────────────────────────
_paddle_ocr_module = None
_easyocr_module = None


def _import_paddleocr():
    global _paddle_ocr_module
    if _paddle_ocr_module is None:
        from paddleocr import PaddleOCR
        _paddle_ocr_module = PaddleOCR
    return _paddle_ocr_module


def _import_easyocr():
    global _easyocr_module
    if _easyocr_module is None:
        import easyocr
        _easyocr_module = easyocr
    return _easyocr_module


# ─── Helper Functions ────────────────────────────────────────────────────────


def _load_image(image_path: str):
    """Load image, handle corrupt/missing files. Returns numpy array or None."""
    try:
        if not os.path.exists(image_path):
            return None
        img = cv2.imread(image_path)
        if img is None:
            return None
        return img
    except Exception:
        return None


def _get_detections(results) -> list:
    """
    Extract detections from YOLO results.
    Returns list of dicts: [{bbox, class_id, class_name, conf}, ...]
    bbox format: [x1, y1, x2, y2] in pixel coordinates
    """
    detections = []
    if results is None or len(results) == 0:
        return detections

    result = results[0]
    if result.boxes is None or len(result.boxes) == 0:
        return detections

    boxes = result.boxes
    for i in range(len(boxes)):
        det = {
            "bbox": boxes.xyxy[i].cpu().numpy().tolist(),  # [x1, y1, x2, y2]
            "class_id": int(boxes.cls[i].cpu().item()),
            "conf": float(boxes.conf[i].cpu().item()),
        }
        detections.append(det)

    return detections


def _bbox_center(bbox: list) -> tuple:
    """Get center point of a bbox [x1, y1, x2, y2]."""
    return ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)


def _center_inside_box(center: tuple, box: list) -> bool:
    """Check if a center point (cx, cy) falls inside a bbox [x1, y1, x2, y2]."""
    cx, cy = center
    return box[0] <= cx <= box[2] and box[1] <= cy <= box[3]


def _bbox_area(bbox: list) -> float:
    """Calculate area of bbox [x1, y1, x2, y2]."""
    return max(0, bbox[2] - bbox[0]) * max(0, bbox[3] - bbox[1])


def _distance_between_centers(bbox1: list, bbox2: list) -> float:
    """Euclidean distance between centers of two bboxes."""
    c1 = _bbox_center(bbox1)
    c2 = _bbox_center(bbox2)
    return ((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2) ** 0.5


def _relative_position_in_box(point: tuple, box: list) -> float:
    """
    Returns vertical relative position of point within box.
    0.0 = top of box, 1.0 = bottom of box.
    """
    if box[3] - box[1] == 0:
        return 0.5
    return (point[1] - box[1]) / (box[3] - box[1])


def _associate_detections_to_bikes(bikes: list, helmets: list,
                                    no_helmets: list, plates: list) -> list:
    """
    Associate helmet/no-helmet/plate detections to bike bounding boxes.

    For each bike:
    - Find helmet/no-helmet with center inside bike bbox (upper 70%)
    - Find plate with center inside bike bbox (lower 60%)
    - Handle overlapping bikes: assign to nearest bike center

    Returns list of dicts:
    [{
        "bike_bbox": [...],
        "num_riders": int,
        "helmet_violations": int,
        "plate_bbox": [...] or None,
        "helmets": [...],
        "no_helmets": [...]
    }]
    """
    if not bikes:
        return []

    # Track which detections are already assigned (for overlapping bikes)
    assigned_helmets = set()
    assigned_no_helmets = set()
    assigned_plates = set()

    vehicle_groups = []

    # Sort bikes by area (largest first) — larger bikes get priority
    bike_indices = sorted(range(len(bikes)), key=lambda i: _bbox_area(bikes[i]["bbox"]), reverse=True)

    for bike_idx in bike_indices:
        bike = bikes[bike_idx]
        bike_bbox = bike["bbox"]
        bike_center = _bbox_center(bike_bbox)

        group = {
            "bike_bbox": bike_bbox,
            "bike_conf": bike["conf"],
            "helmets": [],
            "no_helmets": [],
            "plate_bbox": None,
            "plate_conf": 0.0,
        }

        # ── Find helmets inside this bike bbox ──
        for h_idx, helmet in enumerate(helmets):
            if h_idx in assigned_helmets:
                continue

            h_center = _bbox_center(helmet["bbox"])

            if not _center_inside_box(h_center, bike_bbox):
                continue

            # Geometric prior: helmet should be in upper 70% of bike bbox
            rel_pos = _relative_position_in_box(h_center, bike_bbox)
            if rel_pos > 0.7:
                continue

            # If this helmet is closer to another bike, skip
            if len(bikes) > 1:
                distances = []
                for other_idx in bike_indices:
                    other_bbox = bikes[other_idx]["bbox"]
                    if _center_inside_box(h_center, other_bbox):
                        distances.append((other_idx, _distance_between_centers(helmet["bbox"], other_bbox)))
                if distances:
                    closest = min(distances, key=lambda x: x[1])
                    if closest[0] != bike_idx:
                        continue

            group["helmets"].append(helmet)
            assigned_helmets.add(h_idx)

        # ── Find no-helmets inside this bike bbox ──
        for nh_idx, no_helmet in enumerate(no_helmets):
            if nh_idx in assigned_no_helmets:
                continue

            nh_center = _bbox_center(no_helmet["bbox"])

            if not _center_inside_box(nh_center, bike_bbox):
                continue

            # Geometric prior: no-helmet head should be in upper 70%
            rel_pos = _relative_position_in_box(nh_center, bike_bbox)
            if rel_pos > 0.7:
                continue

            # Nearest bike check
            if len(bikes) > 1:
                distances = []
                for other_idx in bike_indices:
                    other_bbox = bikes[other_idx]["bbox"]
                    if _center_inside_box(nh_center, other_bbox):
                        distances.append((other_idx, _distance_between_centers(no_helmet["bbox"], other_bbox)))
                if distances:
                    closest = min(distances, key=lambda x: x[1])
                    if closest[0] != bike_idx:
                        continue

            group["no_helmets"].append(no_helmet)
            assigned_no_helmets.add(nh_idx)

        # ── Find plate inside this bike bbox ──
        best_plate = None
        best_plate_idx = None
        best_plate_score = -1

        for p_idx, plate in enumerate(plates):
            if p_idx in assigned_plates:
                continue

            p_center = _bbox_center(plate["bbox"])

            if not _center_inside_box(p_center, bike_bbox):
                continue

            # Geometric prior: plate should be in lower 60% of bike bbox
            rel_pos = _relative_position_in_box(p_center, bike_bbox)
            if rel_pos < 0.4:
                continue

            # Score by confidence * proximity
            dist = _distance_between_centers(plate["bbox"], bike_bbox)
            score = plate["conf"] / max(dist, 1.0)

            if score > best_plate_score:
                best_plate_score = score
                best_plate = plate
                best_plate_idx = p_idx

        if best_plate is not None:
            group["plate_bbox"] = best_plate["bbox"]
            group["plate_conf"] = best_plate["conf"]
            assigned_plates.add(best_plate_idx)

        # ── Compute rider count ──
        num_riders = len(group["helmets"]) + len(group["no_helmets"])
        # Sanity cap: no real bike has more than 5 riders
        num_riders = min(num_riders, 5)

        group["num_riders"] = num_riders
        group["helmet_violations"] = len(group["no_helmets"])

        vehicle_groups.append(group)

    return vehicle_groups


def _is_violation(group: dict) -> bool:
    """Check if a vehicle group constitutes a violation."""
    return group["num_riders"] > 2 or group["helmet_violations"] > 0


def _crop_plate(image, plate_bbox: list, padding: float = 0.15):
    """
    Crop plate region with padding.
    Returns cropped numpy array or None.
    """
    h, w = image.shape[:2]
    x1, y1, x2, y2 = plate_bbox

    # Add padding
    pw = (x2 - x1) * padding
    ph = (y2 - y1) * padding
    x1 = max(0, int(x1 - pw))
    y1 = max(0, int(y1 - ph))
    x2 = min(w, int(x2 + pw))
    y2 = min(h, int(y2 + ph))

    if x2 <= x1 or y2 <= y1:
        return None

    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        return None

    return crop


def _preprocess_plate(crop):
    """
    Preprocess plate crop for OCR.
    Resize → grayscale → CLAHE → mild Gaussian blur.
    Returns preprocessed crop.
    """
    if crop is None:
        return None

    # Resize to standard height maintaining aspect ratio
    target_height = 64
    h, w = crop.shape[:2]
    if h == 0:
        return None
    scale = target_height / h
    new_w = max(1, int(w * scale))
    crop = cv2.resize(crop, (new_w, target_height), interpolation=cv2.INTER_CUBIC)

    # Convert to grayscale
    if len(crop.shape) == 3:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    else:
        gray = crop

    # Apply CLAHE for contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # Mild Gaussian blur to reduce noise/compression artifacts
    enhanced = cv2.GaussianBlur(enhanced, (3, 3), 0)

    return enhanced


def _rotate_crop(crop, angle_degrees: float):
    """Rotate plate crop by small angle for OCR retry."""
    if crop is None:
        return None
    h, w = crop.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle_degrees, 1.0)
    rotated = cv2.warpAffine(crop, matrix, (w, h),
                              borderMode=cv2.BORDER_REPLICATE)
    return rotated


def _run_paddle_ocr(ocr_engine, crop) -> tuple:
    """
    Run PaddleOCR on a plate crop.
    Returns (text, confidence) or ("", 0.0) on failure.
    """
    try:
        if crop is None or ocr_engine is None:
            return "", 0.0

        # PaddleOCR expects BGR or grayscale numpy array
        result = ocr_engine.ocr(crop)

        if result is None or len(result) == 0 or result[0] is None:
            return "", 0.0

        # Collect all text lines
        texts = []
        confs = []
        for line in result[0]:
            if line and len(line) >= 2:
                text = line[1][0]
                conf = line[1][1]
                texts.append(text)
                confs.append(conf)

        if not texts:
            return "", 0.0

        full_text = " ".join(texts)
        avg_conf = sum(confs) / len(confs)
        return full_text, avg_conf

    except Exception:
        return "", 0.0


def _run_easy_ocr(ocr_engine, crop) -> tuple:
    """
    Run EasyOCR on a plate crop.
    Returns (text, confidence) or ("", 0.0) on failure.
    """
    try:
        if crop is None:
            return "", 0.0

        result = ocr_engine.readtext(crop)

        if not result:
            return "", 0.0

        texts = []
        confs = []
        for detection in result:
            text = detection[1]
            conf = detection[2]
            texts.append(text)
            confs.append(conf)

        if not texts:
            return "", 0.0

        full_text = " ".join(texts)
        avg_conf = sum(confs) / len(confs)
        return full_text, avg_conf

    except Exception:
        return "", 0.0


def _run_dual_ocr(paddle_ocr, easy_ocr, crop) -> str:
    """
    Run dual OCR strategy:
    1. PaddleOCR first
    2. EasyOCR only if PaddleOCR confidence < 0.3
    3. Rotation retry if both fail
    4. Return best available text (even low-conf beats "")

    Returns cleaned plate text string.
    """
    if crop is None:
        return ""

    # Step 1: PaddleOCR
    paddle_text, paddle_conf = _run_paddle_ocr(paddle_ocr, crop)
    if paddle_conf > 0.3:
        return _clean_plate_text(paddle_text)

    # Step 2: EasyOCR fallback
    easy_text, easy_conf = _run_easy_ocr(easy_ocr, crop)
    if easy_conf > 0.3:
        return _clean_plate_text(easy_text)

    # Step 3: Rotation retry with PaddleOCR
    best_text = paddle_text if paddle_conf >= easy_conf else easy_text
    best_conf = max(paddle_conf, easy_conf)

    for angle in [-5, 5, -10, 10]:
        rotated = _rotate_crop(crop, angle)
        rot_text, rot_conf = _run_paddle_ocr(paddle_ocr, rotated)
        if rot_conf > best_conf:
            best_text = rot_text
            best_conf = rot_conf
        if rot_conf > 0.5:
            return _clean_plate_text(rot_text)

    # Step 4: Return best available (even low-conf beats empty string)
    return _clean_plate_text(best_text)


def _clean_plate_text(raw_text: str) -> str:
    """
    Clean OCR output for license plate text.
    - Strip whitespace
    - Remove special characters (keep alphanumeric, hyphen, space)
    - Apply common OCR error corrections conservatively
    """
    if not raw_text:
        return ""

    text = raw_text.strip()
    text = text.upper()

    # Remove characters that are never in license plates
    cleaned = []
    for ch in text:
        if ch.isalnum() or ch in ("-", " "):
            cleaned.append(ch)
    text = "".join(cleaned)

    # Remove excessive whitespace
    text = " ".join(text.split())

    return text


def _build_output(violation_groups: list) -> dict:
    """Build the output JSON dict from violation groups."""
    violations = []
    for group in violation_groups:
        violations.append({
            "num_riders": int(group["num_riders"]),
            "helmet_violations": int(group["helmet_violations"]),
            "license_plate": str(group.get("plate_text", "")),
        })
    return {"violations": violations}


# ─── Main Class ──────────────────────────────────────────────────────────────


class TrafficViolationDetector:
    """
    Traffic violation detection pipeline for two-wheelers.

    Detects: more than 2 riders, riders without helmets.
    Reports: rider count, helmet violations, license plate text.
    """

    def __init__(self, model_dir: str = "./models"):
        """
        Initialize and load all models.

        Args:
            model_dir: Path to directory containing model weights.
        """
        # ── Device selection ──
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # ── Adaptive image size ──
        self.imgsz = 1280 if torch.cuda.is_available() else 640

        # ── YOLO26s detector ──
        yolo_path = os.path.join(model_dir, "helmet_detector.pt")
        print("Loading YOLO...")
        self.yolo = YOLO(yolo_path)
        print("YOLO loaded")
        # ── PaddleOCR (offline mode) ──
        # try:
        #     PaddleOCR = _import_paddleocr()
        #     # self.paddle_ocr = PaddleOCR(
        #     #     use_textline_orientation=True,
        #     #     use_doc_orientation_classify=False,
        #     #     use_doc_unwarping=False,
        #     #     lang="en",
        #     #     det_model_dir=os.path.join(model_dir, "ocr_det"),
        #     #     rec_model_dir=os.path.join(model_dir, "ocr_rec"),
        #     #     textline_orientation_model_dir=os.path.join(model_dir, "ocr_textline_ori"),
        #     # )
        #     # Replace the current PaddleOCR init with:
        #     print("Loading PaddleOCR...")
        #     self.paddle_ocr = PaddleOCR(
        #         use_angle_cls=True,
        #         lang="en",
        #         # use_gpu=torch.cuda.is_available(),
        #         # show_log=False,
        #     )
        #     print("PaddleOCR loaded")
        # except Exception as e:
        #     print(f"  [WARN] PaddleOCR init failed ({e}), falling back to EasyOCR only")
        #     self.paddle_ocr = None

        try:
            PaddleOCR = _import_paddleocr()
            print("Loading PaddleOCR...")
            paddle_dir = os.path.join(model_dir, "paddleocr")
            self.paddle_ocr = PaddleOCR(
                use_angle_cls=True,
                lang="en",
                # use_gpu=torch.cuda.is_available(),
                # show_log=False,
                det_model_dir=os.path.join(paddle_dir, "PP-OCRv5_server_det"),
                rec_model_dir=os.path.join(paddle_dir, "en_PP-OCRv5_mobile_rec"),
                cls_model_dir=os.path.join(paddle_dir, "PP-LCNet_x1_0_textline_ori"),
            )
            print("PaddleOCR loaded")
        except Exception as e:
            print(f"  [WARN] PaddleOCR init failed ({e}), falling back to EasyOCR only")
            self.paddle_ocr = None

        # ── EasyOCR (offline mode) ──
        easyocr = _import_easyocr()
        print("Loading EasyOCR...")
        self.easy_ocr = easyocr.Reader(
            ["en"],
            gpu=torch.cuda.is_available(),
            model_storage_directory=os.path.join(model_dir, "easyocr_models"),
            download_enabled=False,
        )
        print("EasyOCR loaded")

        # ── Detection thresholds ──
        self.conf_threshold = 0.25
        self.iou_threshold = 0.5

        # ── Class ID mapping (must match data.yaml from training) ──
        # Update these if your data.yaml has different class order
        self.CLASS_BIKE = 0
        self.CLASS_HELMET = 1
        self.CLASS_NO_HELMET = 2
        self.CLASS_PLATE = 3

    def predict(self, image_path: str) -> dict:
        """
        Detect traffic violations in an image.

        Args:
            image_path: Path to input image.

        Returns:
            {
                "violations": [
                    {
                        "num_riders": int,
                        "helmet_violations": int,
                        "license_plate": "string"
                    }
                ]
            }
        """
        try:
            # ── Step 1: Load image ──
            image = _load_image(image_path)
            if image is None:
                return {"violations": []}

            # ── Step 2: Run YOLO detection ──
            results = self.yolo(
                image,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                imgsz=self.imgsz,
                verbose=False,
            )
            detections = _get_detections(results)

            if not detections:
                return {"violations": []}

            # ── Step 3: Separate detections by class ──
            bikes = [d for d in detections if d["class_id"] == self.CLASS_BIKE]
            helmets = [d for d in detections if d["class_id"] == self.CLASS_HELMET]
            no_helmets = [d for d in detections if d["class_id"] == self.CLASS_NO_HELMET]
            plates = [d for d in detections if d["class_id"] == self.CLASS_PLATE]

            if not bikes:
                return {"violations": []}

            # ── Step 4: Associate detections to bikes ──
            vehicle_groups = _associate_detections_to_bikes(
                bikes, helmets, no_helmets, plates
            )

            # ── Step 5: Filter for violations only ──
            violating_groups = [g for g in vehicle_groups if _is_violation(g)]

            if not violating_groups:
                return {"violations": []}

            # ── Step 6: OCR on violating bikes' plates ──
            for group in violating_groups:
                if group["plate_bbox"] is not None:
                    crop = _crop_plate(image, group["plate_bbox"])
                    preprocessed = _preprocess_plate(crop)
                    plate_text = _run_dual_ocr(
                        self.paddle_ocr, self.easy_ocr, preprocessed
                    )
                    group["plate_text"] = plate_text
                else:
                    group["plate_text"] = ""

            # ── Step 7: Build output ──
            return _build_output(violating_groups)

        except Exception:
            return {"violations": []}
