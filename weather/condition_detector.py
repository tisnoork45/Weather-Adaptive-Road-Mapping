from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


# ─────────────────────────────────────────────
#  CONDITION ENUM
# ─────────────────────────────────────────────

class Condition(Enum):
    CLEAR     = auto()
    NORMAL    = auto()
    DARK      = auto()
    NIGHT     = auto()
    RAIN      = auto()
    HEAVY_RAIN= auto()
    FOG       = auto()
    DENSE_FOG = auto()
    SNOW      = auto()
    GLARE     = auto()
    BLUR      = auto()
    DUST      = auto()
    TWILIGHT  = auto()


# Sensor weights: (camera, lidar, radar)
CONDITION_SENSOR_WEIGHTS: Dict[Condition, Tuple[float, float, float]] = {
    Condition.CLEAR:      (0.65, 0.25, 0.10),
    Condition.NORMAL:     (0.50, 0.35, 0.15),
    Condition.DARK:       (0.30, 0.50, 0.20),
    Condition.NIGHT:      (0.20, 0.55, 0.25),
    Condition.RAIN:       (0.20, 0.40, 0.40),
    Condition.HEAVY_RAIN: (0.10, 0.35, 0.55),
    Condition.FOG:        (0.15, 0.35, 0.50),
    Condition.DENSE_FOG:  (0.05, 0.30, 0.65),
    Condition.SNOW:       (0.20, 0.45, 0.35),
    Condition.GLARE:      (0.35, 0.40, 0.25),
    Condition.BLUR:       (0.15, 0.55, 0.30),
    Condition.DUST:       (0.20, 0.40, 0.40),
    Condition.TWILIGHT:   (0.40, 0.40, 0.20),
}

# Human-readable severity labels
CONDITION_SEVERITY: Dict[Condition, str] = {
    Condition.CLEAR:      "GOOD",
    Condition.NORMAL:     "GOOD",
    Condition.TWILIGHT:   "MODERATE",
    Condition.DARK:       "MODERATE",
    Condition.GLARE:      "MODERATE",
    Condition.BLUR:       "MODERATE",
    Condition.RAIN:       "POOR",
    Condition.SNOW:       "POOR",
    Condition.DUST:       "POOR",
    Condition.NIGHT:      "POOR",
    Condition.HEAVY_RAIN: "SEVERE",
    Condition.FOG:        "SEVERE",
    Condition.DENSE_FOG:  "CRITICAL",
}

# Recommended speed reduction factor per condition
CONDITION_SPEED_FACTOR: Dict[Condition, float] = {
    Condition.CLEAR:      1.00,
    Condition.NORMAL:     1.00,
    Condition.TWILIGHT:   0.90,
    Condition.DARK:       0.80,
    Condition.GLARE:      0.75,
    Condition.BLUR:       0.70,
    Condition.RAIN:       0.65,
    Condition.NIGHT:      0.70,
    Condition.SNOW:       0.50,
    Condition.DUST:       0.55,
    Condition.HEAVY_RAIN: 0.40,
    Condition.FOG:        0.35,
    Condition.DENSE_FOG:  0.20,
}


# ─────────────────────────────────────────────
#  RESULT DATA CLASS
# ─────────────────────────────────────────────

@dataclass
class ConditionResult:
    primary: Condition                    # dominant condition
    all_conditions: List[Condition]       # all active conditions
    scores: Dict[str, float]             # raw score per detector
    confidence: float                    # overall confidence 0–1
    sensor_weights: Tuple[float, float, float]  # (cam, lidar, radar)
    severity: str                        # GOOD / MODERATE / POOR / SEVERE / CRITICAL
    speed_factor: float                  # recommended speed multiplier
    visibility_m: float                  # estimated visibility in metres
    brightness: float                    # mean luminance 0–255
    fog_score: float
    rain_score: float
    blur_score: float
    snow_score: float
    glare_score: float
    timestamp: float = field(default_factory=time.time)

    def label(self) -> str:
        return self.primary.name

    def to_dict(self) -> Dict:
        return {
            "primary": self.primary.name,
            "conditions": [c.name for c in self.all_conditions],
            "severity": self.severity,
            "speed_factor": round(self.speed_factor, 2),
            "visibility_m": round(self.visibility_m, 1),
            "brightness": round(self.brightness, 1),
            "confidence": round(self.confidence, 3),
            "weights": {
                "camera": round(self.sensor_weights[0], 2),
                "lidar":  round(self.sensor_weights[1], 2),
                "radar":  round(self.sensor_weights[2], 2),
            },
            "scores": {k: round(v, 3) for k, v in self.scores.items()},
        }


# ─────────────────────────────────────────────
#  MAIN DETECTOR
# ─────────────────────────────────────────────

class ConditionDetector:
    """
    Industry-grade multi-label weather & scene condition detector.

    All detection runs on CPU with pure OpenCV / NumPy — no DL required.

    Parameters
    ----------
    temporal_alpha : float
        EMA smoothing factor for temporal consistency (0 = no smoothing,
        1 = no update). Recommended: 0.35–0.55.
    roi_sky_fraction : float
        Top fraction of frame used for sky/fog analysis.
    roi_road_fraction : float
        Bottom fraction of frame used for road analysis.
    """

    def __init__(
        self,
        temporal_alpha: float = 0.45,
        roi_sky_fraction: float = 0.30,
        roi_road_fraction: float = 0.40,
    ):
        self._alpha = temporal_alpha
        self._sky_frac = roi_sky_fraction
        self._road_frac = roi_road_fraction

        # EMA state for each score
        self._ema: Dict[str, float] = {
            "brightness": 128.0,
            "fog":   0.0,
            "rain":  0.0,
            "blur":  0.0,
            "snow":  0.0,
            "glare": 0.0,
            "dust":  0.0,
        }
        self._frame_count = 0
        print("[ConditionDetector] Initialised — temporal_alpha={:.2f}".format(temporal_alpha))

    # ─────────────────────────────────────────
    #  PUBLIC API
    # ─────────────────────────────────────────

    def detect(self, frame: np.ndarray) -> ConditionResult:
        """
        Analyse a single BGR frame and return full condition report.
        Applies temporal EMA smoothing internally.
        """
        self._frame_count += 1
        img_h, img_w = frame.shape[:2]

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        sky_y = int(img_h * self._sky_frac)
        road_y = int(img_h * (1.0 - self._road_frac))

        sky_roi   = frame[:sky_y, :]
        road_roi  = frame[road_y:, :]
        sky_gray  = gray[:sky_y, :]
        road_gray = gray[road_y:, :]

        # ── Raw scores ──
        brightness  = self._brightness(gray)
        fog_score   = self._fog_score(frame, sky_roi)
        rain_score  = self._rain_score(gray, road_gray)
        blur_score  = self._blur_score(gray)
        snow_score  = self._snow_score(frame, road_roi)
        glare_score = self._glare_score(frame)
        dust_score  = self._dust_score(frame)

        # ── EMA smoothing ──
        brightness  = self._ema_update("brightness", brightness)
        fog_score   = self._ema_update("fog",   fog_score)
        rain_score  = self._ema_update("rain",  rain_score)
        blur_score  = self._ema_update("blur",  blur_score)
        snow_score  = self._ema_update("snow",  snow_score)
        glare_score = self._ema_update("glare", glare_score)
        dust_score  = self._ema_update("dust",  dust_score)

        scores = {
            "brightness": round(brightness, 2),
            "fog":   round(fog_score,   3),
            "rain":  round(rain_score,  3),
            "blur":  round(blur_score,  3),
            "snow":  round(snow_score,  3),
            "glare": round(glare_score, 3),
            "dust":  round(dust_score,  3),
        }

        # ── Active conditions ──
        active: List[Condition] = []

        # Night / dark / twilight
        if brightness < 50:
    active.append(Condition.NIGHT)
elif brightness < 85:
    active.append(Condition.DARK)
elif brightness < 120:
    active.append(Condition.TWILIGHT)

        # Fog
        if fog_score > 0.70:
            active.append(Condition.DENSE_FOG)
        elif fog_score > 0.40:
            active.append(Condition.FOG)

        # Rain
        if rain_score > 0.65:
            active.append(Condition.HEAVY_RAIN)
        elif rain_score > 0.35:
            active.append(Condition.RAIN)

        # Blur
        if blur_score > 0.75:
            active.append(Condition.BLUR)

        # Snow
        if snow_score > 0.45:
            active.append(Condition.SNOW)

        # Glare
        if glare_score > 0.50:
            active.append(Condition.GLARE)

        # Dust
        if dust_score > 0.40:
            active.append(Condition.DUST)

        # Default
        if not active:
            if 95 <= brightness <= 160:
                active.append(Condition.NORMAL)
            elif brightness > 160:
                active.append(Condition.CLEAR)
            else:
                active.append(Condition.NORMAL)

        # ── Primary condition (highest severity) ──
        severity_rank = {
            Condition.DENSE_FOG:  9,
            Condition.HEAVY_RAIN: 8,
            Condition.FOG:        7,
            Condition.NIGHT:      6,
            Condition.SNOW:       6,
            Condition.DUST:       5,
            Condition.RAIN:       5,
            Condition.DARK:       4,
            Condition.GLARE:      4,
            Condition.BLUR:       3,
            Condition.TWILIGHT:   2,
            Condition.NORMAL:     1,
            Condition.CLEAR:      0,
        }
        primary = max(active, key=lambda c: severity_rank.get(c, 0))

        # ── Visibility estimate ──
        visibility_m = self._estimate_visibility(fog_score, rain_score,
                                                  dust_score, brightness)

        # ── Confidence: higher when scores are unambiguous ──
        max_score = max(fog_score, rain_score, snow_score, glare_score,
                        dust_score, blur_score, 1 - brightness / 255)
        confidence = float(np.clip(0.50 + 0.50 * max_score, 0.50, 0.98))

        weights = CONDITION_SENSOR_WEIGHTS.get(primary,
                  CONDITION_SENSOR_WEIGHTS[Condition.NORMAL])
        severity  = CONDITION_SEVERITY.get(primary, "MODERATE")
        spd_factor = CONDITION_SPEED_FACTOR.get(primary, 0.80)

        return ConditionResult(
            primary=primary,
            all_conditions=active,
            scores=scores,
            confidence=round(confidence, 3),
            sensor_weights=weights,
            severity=severity,
            speed_factor=spd_factor,
            visibility_m=visibility_m,
            brightness=brightness,
            fog_score=fog_score,
            rain_score=rain_score,
            blur_score=blur_score,
            snow_score=snow_score,
            glare_score=glare_score,
        )

    # ─────────────────────────────────────────
    #  INDIVIDUAL DETECTORS
    # ─────────────────────────────────────────

    @staticmethod
    def _brightness(gray: np.ndarray) -> float:
        return float(np.mean(gray))

    # ── FOG — Dark Channel Prior (He et al. 2009) ──
    @staticmethod
    def _fog_score(frame: np.ndarray, sky_roi: np.ndarray) -> float:
        """
        Fog increases the dark channel value (min of RGB in local patch).
        Hazy images have a high dark channel due to atmospheric scattering.
        Also checks contrast reduction in the sky region.
        """
        # Dark channel on downsampled frame for speed
        small = cv2.resize(frame, (160, 90))
        dark = np.min(small, axis=2)
        # Min-filter (erosion) 15×15 patch
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        dark_channel = cv2.erode(dark.astype(np.float32), kernel)
        dark_mean = float(np.mean(dark_channel)) / 255.0

        # Sky contrast: foggy sky has very low contrast
        sky_gray = cv2.cvtColor(sky_roi, cv2.COLOR_BGR2GRAY) \
            if sky_roi.size > 0 else np.zeros((1,), dtype=np.uint8)
        sky_std = float(np.std(sky_gray)) / 128.0

        # Combine: high dark channel + low sky contrast → fog
        fog = dark_mean * 0.65 + max(0.0, 0.35 * (1.0 - sky_std * 2.0))
        return float(np.clip(fog, 0.0, 1.0))

    # ── RAIN — streak + texture ──
    @staticmethod
    def _rain_score(gray: np.ndarray, road_gray: np.ndarray) -> float:
        """
        Rain streaks appear as near-vertical high-frequency edges.
        Uses Sobel vertical edge energy + FFT vertical energy ratio.
        """
        small = cv2.resize(gray, (160, 90))

        # Sobel vertical (rain streaks are mostly vertical)
        sobel_v = cv2.Sobel(small, cv2.CV_64F, 0, 1, ksize=3)
        sobel_h = cv2.Sobel(small, cv2.CV_64F, 1, 0, ksize=3)
        v_energy = float(np.mean(np.abs(sobel_v)))
        h_energy = float(np.mean(np.abs(sobel_h))) + 1e-6
        streak_ratio = v_energy / h_energy

        # FFT: rain introduces high-frequency vertical energy
        f = np.fft.fft2(small.astype(np.float32))
        fshift = np.abs(np.fft.fftshift(f))
        h, w = fshift.shape
        # Horizontal band (vertical frequency)
        vert_band = fshift[h//2-5:h//2+5, :]
        horiz_band = fshift[:, w//2-5:w//2+5]
        fft_ratio = float(np.mean(vert_band)) / (float(np.mean(horiz_band)) + 1e-6)

        # Road wetness: high specular reflection variance
        road_std = float(np.std(road_gray)) / 128.0 if road_gray.size > 0 else 0.0

        rain = (
            np.clip((streak_ratio - 1.0) / 3.0, 0, 1) * 0.40 +
            np.clip((fft_ratio  - 1.0) / 3.0, 0, 1) * 0.35 +
            np.clip(road_std, 0, 1) * 0.25
        )
        return float(np.clip(rain, 0.0, 1.0))

    # ── BLUR — Laplacian + FFT high-freq energy ──
    @staticmethod
    def _blur_score(gray: np.ndarray) -> float:
        """
        Laplacian variance drops sharply for blurry images.
        FFT energy in high-frequency ring also decreases.
        """
        lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        # Normalise: <10 = very blurry, >500 = sharp
        lap_score = 1.0 - float(np.clip(lap_var / 500.0, 0.0, 1.0))

        # FFT high-freq energy ratio
        small = cv2.resize(gray, (160, 90)).astype(np.float32)
        f = np.abs(np.fft.fftshift(np.fft.fft2(small)))
        h, w = f.shape
        cy, cx = h // 2, w // 2
        # High-frequency ring mask
        Y, X = np.ogrid[:h, :w]
        dist = np.sqrt((X - cx)**2 + (Y - cy)**2)
        hf_mask = dist > min(h, w) * 0.35
        hf_energy = float(np.mean(f[hf_mask]))
        total_energy = float(np.mean(f)) + 1e-6
        hf_ratio = 1.0 - float(np.clip(hf_energy / total_energy / 2.0, 0.0, 1.0))

        blur = lap_score * 0.70 + hf_ratio * 0.30
        return float(np.clip(blur, 0.0, 1.0))

    # ── SNOW — white pixel ratio + texture uniformity ──
    @staticmethod
    def _snow_score(frame: np.ndarray, road_roi: np.ndarray) -> float:
        """
        Snow appears as high-brightness, low-saturation regions.
        Uses HSV saturation + value channels.
        """
        small = cv2.resize(frame, (160, 90))
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        s = hsv[:, :, 1].astype(float) / 255.0   # saturation
        v = hsv[:, :, 2].astype(float) / 255.0   # value (brightness)

        # Snow: high V, low S
        snow_mask = (v > 0.75) & (s < 0.20)
        white_ratio = float(np.mean(snow_mask))

        # Texture uniformity on road (snow flattens texture)
        if road_roi.size > 0:
            road_small = cv2.resize(road_roi, (80, 45))
            road_gray  = cv2.cvtColor(road_small, cv2.COLOR_BGR2GRAY)
            texture_std = float(np.std(road_gray)) / 128.0
            texture_score = max(0.0, 1.0 - texture_std * 2.0)
        else:
            texture_score = 0.0

        snow = white_ratio * 0.60 + texture_score * 0.40
        return float(np.clip(snow, 0.0, 1.0))

    # ── GLARE — highlight saturation ──
    @staticmethod
    def _glare_score(frame: np.ndarray) -> float:
        """
        Glare: large over-exposed (saturated white) regions.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, bright_mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
        glare_ratio = float(np.mean(bright_mask > 0))
        return float(np.clip(glare_ratio * 5.0, 0.0, 1.0))

    # ── DUST / SANDSTORM — yellowish low-contrast cast ──
    @staticmethod
    def _dust_score(frame: np.ndarray) -> float:
        """
        Dust/sandstorm: yellowish-brownish cast + very low contrast.
        Checks hue distribution and global contrast.
        """
        small = cv2.resize(frame, (160, 90))
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        h_ch = hsv[:, :, 0].astype(float)
        s_ch = hsv[:, :, 1].astype(float) / 255.0

        # Hue 15–35° = yellow-orange (dust range in OpenCV scale: 8–18)
        yellow_mask = (h_ch >= 8) & (h_ch <= 25) & (s_ch > 0.15)
        yellow_ratio = float(np.mean(yellow_mask))

        # Low contrast indicator
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        contrast = float(np.std(gray)) / 128.0
        low_contrast = max(0.0, 1.0 - contrast * 3.0)

        dust = yellow_ratio * 0.55 + low_contrast * 0.45
        return float(np.clip(dust, 0.0, 1.0))

    # ─────────────────────────────────────────
    #  VISIBILITY ESTIMATION
    # ─────────────────────────────────────────

    @staticmethod
    def _estimate_visibility(
        fog: float, rain: float, dust: float, brightness: float
    ) -> float:
        """
        Empirical visibility model in metres.
        Based on Koschmieder's law for fog and adaptation for rain/dust.
        """
        # Fog: Koschmieder ~  visibility = 3.912 / extinction_coefficient
        # Map fog_score 0→1 to ext coeff 0.01→0.5
        if fog > 0.05:
            ext = 0.01 + fog * 0.49
            vis_fog = 3.912 / ext
        else:
            vis_fog = 10000.0

        # Rain: empirical (Kunkel 1984)
        if rain > 0.05:
            rain_rate_mm_h = rain * 50.0   # rough mapping
            vis_rain = 1000.0 / (0.05 + 0.03 * rain_rate_mm_h)
        else:
            vis_rain = 10000.0

        # Dust
        if dust > 0.05:
            vis_dust = 500.0 * (1.0 - dust)
        else:
            vis_dust = 10000.0

        # Night
        vis_night = 10000.0
        if brightness < 60:
            vis_night = 100.0 + brightness * 3.0

        vis = min(vis_fog, vis_rain, vis_dust, vis_night)
        return float(np.clip(vis, 5.0, 10000.0))

    # ─────────────────────────────────────────
    #  EMA HELPER
    # ─────────────────────────────────────────

    def _ema_update(self, key: str, new_val: float) -> float:
        prev = self._ema.get(key, new_val)
        smoothed = self._alpha * prev + (1.0 - self._alpha) * new_val
        self._ema[key] = smoothed
        return smoothed


    def ema_state(self) -> Dict[str, float]:
        return dict(self._ema)

    @property
    def frame_count(self) -> int:
        return self._frame_count
