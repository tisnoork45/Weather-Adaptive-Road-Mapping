import time
import warnings
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch

warnings.filterwarnings("ignore")

try:
    from ultralytics import YOLO
except ImportError:
    raise ImportError("Install ultralytics: pip install ultralytics")


# ─────────────────────────────────────────────
#  DATA CLASSES
# ─────────────────────────────────────────────

@dataclass
class Detection:
    """Single detection result with all metadata."""
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float
    class_id: int
    class_name: str
    category: str                        # ADAS category
    box_w: int = 0
    box_h: int = 0
    cx: float = 0.0                      # centre x (normalised)
    cy: float = 0.0                      # centre y (normalised)
    area_ratio: float = 0.0              # box area / frame area
    aspect_ratio: float = 0.0
    is_occluded: bool = False
    occlusion_ratio: float = 0.0
    frame_id: int = 0
    timestamp: float = field(default_factory=time.time)

    def to_tlwh(self) -> Tuple[int, int, int, int]:
        return self.x1, self.y1, self.box_w, self.box_h

    def to_xyxy(self) -> Tuple[int, int, int, int]:
        return self.x1, self.y1, self.x2, self.y2

    def iou(self, other: "Detection") -> float:
        ix1 = max(self.x1, other.x1)
        iy1 = max(self.y1, other.y1)
        ix2 = min(self.x2, other.x2)
        iy2 = min(self.y2, other.y2)
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        union = (self.box_w * self.box_h +
                 other.box_w * other.box_h - inter)
        return inter / union if union > 0 else 0.0


# ─────────────────────────────────────────────
#  ADAS CLASS TAXONOMY
# ─────────────────────────────────────────────

# COCO class id → ADAS category
ADAS_CATEGORY_MAP: Dict[int, str] = {
    # Vulnerable Road Users
    0:  "pedestrian",
    1:  "cyclist",
    # Two-wheelers
    3:  "motorcycle",
    # Light vehicles
    2:  "car",
    5:  "bus",
    7:  "truck",
    # Non-motorised
    4:  "airplane",       # edge case
    # Animals (high risk)
    15: "animal",
    16: "animal",
    17: "animal",
    18: "animal",
    19: "animal",
    20: "animal",
    21: "animal",
    22: "animal",
    23: "animal",
    # Road infrastructure
    9:  "traffic_light",
    11: "stop_sign",
    12: "parking_meter",
    # Other obstacles
    24: "backpack",
    25: "umbrella",
    26: "handbag",
    28: "suitcase",
    56: "chair",
    57: "couch",
    59: "bed",
    60: "dining_table",
}

# Priority score per ADAS category (higher = more dangerous)
CATEGORY_PRIORITY: Dict[str, int] = {
    "pedestrian":    10,
    "cyclist":        9,
    "animal":         8,
    "motorcycle":     7,
    "car":            6,
    "truck":          6,
    "bus":            6,
    "traffic_light":  5,
    "stop_sign":      5,
    "parking_meter":  3,
    "backpack":       2,
    "umbrella":       2,
    "handbag":        2,
    "suitcase":       2,
    "chair":          1,
    "other":          1,
}

# Per-class confidence thresholds (tuned for ADAS false-positive tolerance)
PER_CLASS_CONF: Dict[str, float] = {
    "pedestrian":    0.30,
    "cyclist":       0.30,
    "animal":        0.28,
    "motorcycle":    0.30,
    "car":           0.35,
    "truck":         0.35,
    "bus":           0.35,
    "traffic_light": 0.40,
    "stop_sign":     0.40,
}
DEFAULT_CONF_THRESHOLD = 0.25

# Minimum pixel height per class at typical detection distance
MIN_BOX_HEIGHT: Dict[str, int] = {
    "pedestrian": 20,
    "cyclist":    18,
    "car":        15,
    "truck":      20,
    "bus":        20,
    "motorcycle": 15,
    "animal":     12,
}
DEFAULT_MIN_HEIGHT = 10


# ─────────────────────────────────────────────
#  DETECTOR CLASS
# ─────────────────────────────────────────────

class ObjectDetector:
    """
    Industry-grade YOLO detector for ADAS pipelines.

    Parameters
    ----------
    model_path : str
        Path to YOLOv8 .pt weights file.
    input_size : int
        YOLO inference resolution (320 / 640 / 1280).
    use_half : bool
        FP16 inference on CUDA (faster, negligible accuracy drop).
    nms_type : str
        'standard' or 'soft' — Soft-NMS is better for occluded objects.
    roi_bottom_fraction : float
        Fraction of frame height used as ROI (ignore sky region).
        E.g. 0.75 means top 25 % of frame is ignored.
    history_len : int
        Number of past frames kept for temporal smoothing / ghost filtering.
    """

    def __init__(
        self,
        model_path: str = "weights/yolov8n.pt",
        input_size: int = 640,
        use_half: bool = True,
        nms_type: str = "soft",
        roi_bottom_fraction: float = 0.85,
        history_len: int = 5,
    ):
        self.input_size = input_size
        self.nms_type = nms_type
        self.roi_bottom_fraction = roi_bottom_fraction
        self.history_len = history_len
        self.frame_id = 0
        self._detection_history: deque = deque(maxlen=history_len)
        self._inference_times: deque = deque(maxlen=30)

        # ── Device selection ──
        self.device = (
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )
        self.use_half = use_half and self.device == "cuda"

        # ── Load model ──
        print(f"[Detector] Loading model: {model_path}  device={self.device}  "
              f"half={self.use_half}  nms={nms_type}")
        self.model = YOLO(model_path)
        self.model.to(self.device)
        if self.use_half:
            self.model.model.half()

        self.class_names: Dict[int, str] = self.model.names
        print(f"[Detector] Ready — {len(self.class_names)} classes")

    # ─────────────────────────────────────────
    #  PUBLIC API
    # ─────────────────────────────────────────

    def detect(
        self,
        frame: np.ndarray,
        conf_override: Optional[float] = None,
    ) -> List[Detection]:
        """
        Run detection on a single BGR frame.

        Returns a list of Detection objects sorted by priority (most
        dangerous first) then by proximity (largest box first).
        """
        self.frame_id += 1
        img_h, img_w = frame.shape[:2]

        # ── ROI crop (removes sky / irrelevant top region) ──
        roi_y_start = int(img_h * (1.0 - self.roi_bottom_fraction))
        roi_frame = frame[roi_y_start:, :]

        # ── Inference ──
        t0 = time.perf_counter()
        results = self.model(
            roi_frame,
            imgsz=self.input_size,
            conf=conf_override or DEFAULT_CONF_THRESHOLD,
            iou=0.45,
            agnostic_nms=True,
            verbose=False,
        )[0]
        dt = time.perf_counter() - t0
        self._inference_times.append(dt)

        raw_boxes = results.boxes
        detections: List[Detection] = []

        if raw_boxes is None or len(raw_boxes) == 0:
            self._detection_history.append(detections)
            return detections

        for box in raw_boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            y1 += roi_y_start          # shift back to full-frame coords
            y2 += roi_y_start

            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            cls_name = self.class_names.get(cls_id, "unknown")
            category = ADAS_CATEGORY_MAP.get(cls_id, "other")

            # ── Per-class confidence gate ──
            min_conf = PER_CLASS_CONF.get(category, DEFAULT_CONF_THRESHOLD)
            if conf < min_conf:
                continue

            box_w = x2 - x1
            box_h = y2 - y1

            # ── Minimum size filter ──
            min_h = MIN_BOX_HEIGHT.get(category, DEFAULT_MIN_HEIGHT)
            if box_h < min_h or box_w < 8:
                continue

            # ── Aspect ratio sanity (ignore degenerate boxes) ──
            ar = box_w / max(box_h, 1)
            if ar > 8.0 or ar < 0.1:
                continue

            cx = ((x1 + x2) / 2) / img_w
            cy = ((y1 + y2) / 2) / img_h
            area_ratio = (box_w * box_h) / (img_w * img_h)

            det = Detection(
                x1=x1, y1=y1, x2=x2, y2=y2,
                confidence=conf,
                class_id=cls_id,
                class_name=cls_name,
                category=category,
                box_w=box_w, box_h=box_h,
                cx=cx, cy=cy,
                area_ratio=area_ratio,
                aspect_ratio=ar,
                frame_id=self.frame_id,
            )
            detections.append(det)

        # ── NMS ──
        if self.nms_type == "soft":
            detections = self._soft_nms(detections)
        else:
            detections = self._standard_nms(detections)

        # ── Occlusion estimation ──
        detections = self._estimate_occlusion(detections)

        # ── Temporal ghost filter ──
        detections = self._ghost_filter(detections)

        # ── Sort: priority desc, then area desc ──
        detections.sort(
            key=lambda d: (
                -CATEGORY_PRIORITY.get(d.category, 1),
                -d.area_ratio,
            )
        )

        self._detection_history.append(detections)
        return detections

    # ─────────────────────────────────────────
    #  NMS IMPLEMENTATIONS
    # ─────────────────────────────────────────

    @staticmethod
    def _standard_nms(
        dets: List[Detection],
        iou_thresh: float = 0.45,
    ) -> List[Detection]:
        if not dets:
            return dets
        dets_sorted = sorted(dets, key=lambda d: -d.confidence)
        kept = []
        while dets_sorted:
            best = dets_sorted.pop(0)
            kept.append(best)
            dets_sorted = [
                d for d in dets_sorted
                if best.iou(d) < iou_thresh
            ]
        return kept

    @staticmethod
    def _soft_nms(
        dets: List[Detection],
        iou_thresh: float = 0.45,
        sigma: float = 0.5,
        score_thresh: float = 0.20,
    ) -> List[Detection]:
        """
        Soft-NMS (Bodla et al. 2017): penalises overlapping boxes
        rather than hard-suppressing them. Better for occluded objects.
        """
        if not dets:
            return dets
        dets = sorted(dets, key=lambda d: -d.confidence)
        scores = [d.confidence for d in dets]

        for i in range(len(dets)):
            for j in range(i + 1, len(dets)):
                iou = dets[i].iou(dets[j])
                if iou > iou_thresh:
                    # Gaussian penalty
                    scores[j] *= np.exp(-(iou ** 2) / sigma)

        kept = [d for d, s in zip(dets, scores) if s >= score_thresh]
        for d, s in zip(kept, scores):
            d.confidence = s
        return kept

    # ─────────────────────────────────────────
    #  OCCLUSION ESTIMATION
    # ─────────────────────────────────────────

    @staticmethod
    def _estimate_occlusion(dets: List[Detection]) -> List[Detection]:
        """
        Estimates pairwise occlusion between detections.
        Marks a detection as occluded if another detection overlaps
        it by more than 30 %.
        """
        for i, d in enumerate(dets):
            max_overlap = 0.0
            for j, other in enumerate(dets):
                if i == j:
                    continue
                iou = d.iou(other)
                if iou > max_overlap:
                    max_overlap = iou
            d.occlusion_ratio = max_overlap
            d.is_occluded = max_overlap > 0.30
        return dets

    # ─────────────────────────────────────────
    #  TEMPORAL GHOST FILTER
    # ─────────────────────────────────────────

    def _ghost_filter(self, dets: List[Detection]) -> List[Detection]:
        """
        Removes detections that appear only once in a single frame
        and have very low confidence — likely false positives / ghosts.
        Requires at least 2 frames of history before activating.
        """
        if len(self._detection_history) < 2:
            return dets

        prev_dets = self._detection_history[-1]
        if not prev_dets:
            return dets

        filtered = []
        for d in dets:
            # Always keep high-confidence detections
            if d.confidence >= 0.45:
                filtered.append(d)
                continue
            # Keep if matched a detection in the previous frame
            matched = any(d.iou(p) > 0.25 for p in prev_dets)
            if matched:
                filtered.append(d)
        return filtered

    # ─────────────────────────────────────────
    #  DIAGNOSTICS
    # ─────────────────────────────────────────

    @property
    def avg_inference_ms(self) -> float:
        if not self._inference_times:
            return 0.0
        return float(np.mean(self._inference_times)) * 1000.0

    def get_stats(self) -> Dict:
        return {
            "frame_id": self.frame_id,
            "device": self.device,
            "avg_inference_ms": round(self.avg_inference_ms, 2),
            "fps_estimate": round(1000.0 / max(self.avg_inference_ms, 1), 1),
        }

    def warmup(self, img_h: int = 720, img_w: int = 1280, n: int = 3) -> None:
        """Run n dummy inferences to warm up GPU/model."""
        print(f"[Detector] Warming up ({n} passes)…")
        dummy = np.zeros((img_h, img_w, 3), dtype=np.uint8)
        for _ in range(n):
            self.detect(dummy)
        self.frame_id = 0
        self._detection_history.clear()
        self._inference_times.clear()
        print("[Detector] Warmup complete.")
