from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
@dataclass
class EnhancedFrame:
    image: np.ndarray         # BGR, same size as input
    pipeline: str             # which pipeline was run
    gamma_applied: float
    clahe_applied: bool
    dehaze_applied: bool
    denoise_applied: bool
    processing_ms: float

class ImageEnhancer:
    def __init__(
        self,
        clahe_clip: float = 2.5,
        clahe_tile: int = 8,
        bilateral_d: int = 5,
        dehaze_omega: float = 0.85,
        msr_scales: Optional[list] = None,
    ):
        self._clahe_clip = clahe_clip
        self._clahe_tile = clahe_tile
        self._bilateral_d = bilateral_d
        self._dehaze_omega = dehaze_omega
        self._msr_scales = msr_scales or [15, 80, 250]

        # Pre-build CLAHE objects for different clip limits
        self._clahe_std  = cv2.createCLAHE(clipLimit=clahe_clip,
                                             tileGridSize=(clahe_tile, clahe_tile))
        self._clahe_hard = cv2.createCLAHE(clipLimit=4.0,
                                             tileGridSize=(8, 8))
        self._clahe_soft = cv2.createCLAHE(clipLimit=1.5,
                                             tileGridSize=(16, 16))

    def enhance(
        self,
        frame: np.ndarray,
        condition: str,
    ) -> EnhancedFrame:
        
        t0 = time.perf_counter()
        c = condition.lower()

        gamma_applied  = 1.0
        clahe_applied  = False
        dehaze_applied = False
        denoise_applied= False

        if c in ("night",):
            out, gamma_applied = self._pipeline_night(frame)
            clahe_applied = True
            denoise_applied = True

        elif c in ("dark", "twilight"):
            out, gamma_applied = self._pipeline_dark(frame)
            clahe_applied = True

        elif c in ("dense_fog",):
            out = self._pipeline_dense_fog(frame)
            dehaze_applied = True
            clahe_applied  = True

        elif c in ("fog",):
            out = self._pipeline_fog(frame)
            dehaze_applied = True

        elif c in ("heavy_rain",):
            out = self._pipeline_heavy_rain(frame)
            clahe_applied  = True
            denoise_applied= True

        elif c in ("rain",):
            out = self._pipeline_rain(frame)
            clahe_applied = True

        elif c in ("snow",):
            out = self._pipeline_snow(frame)
            clahe_applied = True

        elif c in ("glare",):
            out = self._pipeline_glare(frame)
            clahe_applied = True

        elif c in ("blur",):
            out, gamma_applied = self._pipeline_blur(frame)
            clahe_applied = True

        elif c in ("dust",):
            out = self._pipeline_dust(frame)
            clahe_applied = True

        elif c in ("clear",):
            # Minimal processing — avoid over-sharpening
            out = self._apply_clahe_ycrcb(frame, self._clahe_soft)
            clahe_applied = True

        else:  # normal
            out = self._apply_clahe_ycrcb(frame, self._clahe_std)
            clahe_applied = True

        dt = (time.perf_counter() - t0) * 1000.0

        return EnhancedFrame(
            image=out,
            pipeline=c,
            gamma_applied=round(gamma_applied, 3),
            clahe_applied=clahe_applied,
            dehaze_applied=dehaze_applied,
            denoise_applied=denoise_applied,
            processing_ms=round(dt, 2),
        )
    def _pipeline_night(
        self, img: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        """Multi-Scale Retinex + CLAHE + bilateral denoise."""
        # MSR for illumination normalisation
        msr = self._multi_scale_retinex(img)
        # Adaptive gamma
        mean_lum = float(np.mean(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)))
        gamma = self._estimate_gamma(mean_lum, target=100.0)
        msr = self._apply_gamma(msr, gamma)
        # CLAHE
        msr = self._apply_clahe_lab(msr, self._clahe_hard)
        # Edge-preserving denoise
        out = cv2.bilateralFilter(msr, self._bilateral_d, 75, 75)
        return out, gamma

    def _pipeline_dark(
        self, img: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        """Gamma lift + CLAHE on LAB."""
        mean_lum = float(np.mean(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)))
        gamma = self._estimate_gamma(mean_lum, target=110.0)
        img = self._apply_gamma(img, gamma)
        out = self._apply_clahe_lab(img, self._clahe_hard)
        return out, gamma

    def _pipeline_fog(self, img: np.ndarray) -> np.ndarray:
        """Dark Channel Prior dehazing + CLAHE."""
        dehazed = self._dark_channel_dehaze(img, omega=self._dehaze_omega)
        out = self._apply_clahe_ycrcb(dehazed, self._clahe_std)
        return out

    def _pipeline_dense_fog(self, img: np.ndarray) -> np.ndarray:
        """Stronger dehazing + sharpening + CLAHE."""
        dehazed = self._dark_channel_dehaze(img, omega=0.95)
        # Unsharp mask to recover edges lost in fog
        sharpened = self._unsharp_mask(dehazed, amount=0.8, sigma=2.0)
        out = self._apply_clahe_lab(sharpened, self._clahe_hard)
        return out

    def _pipeline_rain(self, img: np.ndarray) -> np.ndarray:
        """Rain streak removal + CLAHE."""
        de_rained = self._remove_rain_streaks(img)
        out = self._apply_clahe_ycrcb(de_rained, self._clahe_std)
        return out

    def _pipeline_heavy_rain(self, img: np.ndarray) -> np.ndarray:
        """Stronger rain removal + bilateral + CLAHE."""
        de_rained = self._remove_rain_streaks(img, strength=1.5)
        denoised  = cv2.bilateralFilter(de_rained, 7, 80, 80)
        out = self._apply_clahe_lab(denoised, self._clahe_hard)
        return out

    def _pipeline_snow(self, img: np.ndarray) -> np.ndarray:
        """
        Snow: white-balance correction (snow cast is bluish) +
        percentile contrast stretch + CLAHE.
        """
        wb = self._grey_world_white_balance(img)
        stretched = self._percentile_stretch(wb, lo=1, hi=99)
        out = self._apply_clahe_ycrcb(stretched, self._clahe_std)
        return out

    def _pipeline_glare(self, img: np.ndarray) -> np.ndarray:
        """
        Glare / overexposure: local tone mapping (ACES) + CLAHE.
        """
        tmo = self._aces_tone_map(img)
        out = self._apply_clahe_ycrcb(tmo, self._clahe_soft)
        return out

    def _pipeline_blur(
        self, img: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        """Unsharp masking + mild gamma + CLAHE."""
        sharpened = self._unsharp_mask(img, amount=1.2, sigma=1.5)
        mean_lum  = float(np.mean(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)))
        gamma = self._estimate_gamma(mean_lum, target=120.0)
        sharpened = self._apply_gamma(sharpened, gamma)
        out = self._apply_clahe_ycrcb(sharpened, self._clahe_std)
        return out, gamma

    def _pipeline_dust(self, img: np.ndarray) -> np.ndarray:
        """
        Dust: remove yellowish cast via grey-world WB +
        saturation boost + CLAHE.
        """
        wb = self._grey_world_white_balance(img)
        sat = self._boost_saturation(wb, scale=1.25)
        out = self._apply_clahe_ycrcb(sat, self._clahe_std)
        return out

    @staticmethod
    def _apply_clahe_ycrcb(
        img: np.ndarray, clahe: cv2.CLAHE
    ) -> np.ndarray:
        ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        y, cr, cb = cv2.split(ycrcb)
        ycrcb = cv2.merge((clahe.apply(y), cr, cb))
        return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)

    @staticmethod
    def _apply_clahe_lab(
        img: np.ndarray, clahe: cv2.CLAHE
    ) -> np.ndarray:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        lab = cv2.merge((clahe.apply(l), a, b))
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # ── Gamma ──
    @staticmethod
    def _estimate_gamma(mean_luminance: float, target: float = 110.0) -> float:
        """Estimate gamma to bring mean luminance to target."""
        ratio = target / max(mean_luminance, 1.0)
        gamma = np.log(0.5) / np.log(0.5 / max(ratio, 0.01))
        return float(np.clip(gamma, 0.5, 4.0))

    @staticmethod
    def _apply_gamma(img: np.ndarray, gamma: float) -> np.ndarray:
        if abs(gamma - 1.0) < 0.05:
            return img
        lut = np.array(
            [((i / 255.0) ** (1.0 / gamma)) * 255 for i in range(256)],
            dtype=np.uint8
        )
        return cv2.LUT(img, lut)

    # ── Dark Channel Prior Dehazing (He et al. 2009) ──
    def _dark_channel_dehaze(
        self, img: np.ndarray, omega: float = 0.85,
        patch_size: int = 15, t_min: float = 0.10
    ) -> np.ndarray:
        """
        Full DCP dehazing:
        1. Compute dark channel
        2. Estimate atmospheric light A
        3. Estimate transmission map t
        4. Refine t with guided filter
        5. Recover scene radiance J
        """
        I = img.astype(np.float64) / 255.0
        h, w = I.shape[:2]

        # 1. Dark channel
        dark = self._compute_dark_channel(I, patch_size)

        # 2. Atmospheric light (top 0.1% bright pixels in dark channel)
        flat_dark = dark.ravel()
        flat_idx  = np.argsort(flat_dark)[-max(1, int(len(flat_dark) * 0.001)):]
        bright_pixels = I.reshape(-1, 3)[flat_idx]
        A = np.max(bright_pixels, axis=0)
        A = np.clip(A, 0.5, 1.0)   # prevent over-brightening

        # 3. Transmission estimate
        A_rep = A.reshape(1, 1, 3)
        t_est = 1.0 - omega * self._compute_dark_channel(I / A_rep, patch_size)
        t_est = np.clip(t_est, t_min, 1.0)

        # 4. Guided filter refinement (edge-aware)
        guide = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        t_refined = self._guided_filter(guide, t_est.astype(np.float32),
                                         radius=40, eps=1e-3)
        t_refined = np.clip(t_refined, t_min, 1.0)

        # 5. Scene recovery
        t3 = np.stack([t_refined] * 3, axis=2)
        J  = (I - A_rep) / np.maximum(t3, t_min) + A_rep
        J  = np.clip(J, 0.0, 1.0)

        return (J * 255.0).astype(np.uint8)

    @staticmethod
    def _compute_dark_channel(
        I: np.ndarray, patch_size: int
    ) -> np.ndarray:
        dark_pixel = np.min(I, axis=2)
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (patch_size, patch_size)
        )
        return cv2.erode(dark_pixel.astype(np.float32), kernel)

    @staticmethod
    def _guided_filter(
        guide: np.ndarray,
        src: np.ndarray,
        radius: int = 40,
        eps: float = 1e-3,
    ) -> np.ndarray:
        """
        Fast guided image filter using box filter approximation.
        """
        r = radius
        def box(x):
            return cv2.boxFilter(x, -1, (2*r+1, 2*r+1))

        N = box(np.ones_like(guide))
        mean_I = box(guide)  / N
        mean_p = box(src)    / N
        corr_I = box(guide * guide) / N
        corr_Ip= box(guide * src)   / N

        var_I  = corr_I  - mean_I * mean_I
        cov_Ip = corr_Ip - mean_I * mean_p

        a = cov_Ip / (var_I + eps)
        b = mean_p - a * mean_I

        mean_a = box(a) / N
        mean_b = box(b) / N

        return mean_a * guide + mean_b

    # ── Multi-Scale Retinex ──
    def _multi_scale_retinex(self, img: np.ndarray) -> np.ndarray:
        """
        MSR: log(I) - log(sum of Gaussian blurs at multiple scales).
        Normalises illumination variation.
        """
        img_f = img.astype(np.float32) + 1.0
        msr = np.zeros_like(img_f)
        for sigma in self._msr_scales:
            blurred = cv2.GaussianBlur(img_f, (0, 0), sigma)
            msr += np.log(img_f) - np.log(blurred + 1.0)

        msr /= len(self._msr_scales)
        # Normalise to 0–255
        for c in range(3):
            ch = msr[:, :, c]
            lo, hi = np.percentile(ch, [1, 99])
            msr[:, :, c] = np.clip((ch - lo) / max(hi - lo, 1e-6), 0, 1) * 255.0

        return msr.astype(np.uint8)

    # ── Rain streak removal ──
    def _remove_rain_streaks(
        self, img: np.ndarray, strength: float = 1.0
    ) -> np.ndarray:
        """
        Rain streaks: near-vertical lines in high-frequency domain.
        Method: morphological top-hat → estimate streak map → subtract.
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Vertical structuring element to isolate streaks
        kernel_h = max(3, int(15 * strength))
        se = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_h))
        tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, se)
        # Threshold streak map
        _, streak_mask = cv2.threshold(tophat, 20, 255, cv2.THRESH_BINARY)
        # Inpaint streaks
        out = cv2.inpaint(img, streak_mask, inpaintRadius=2,
                          flags=cv2.INPAINT_TELEA)
        return out

    # ── Unsharp mask ──
    @staticmethod
    def _unsharp_mask(
        img: np.ndarray, amount: float = 1.0, sigma: float = 1.5
    ) -> np.ndarray:
        blurred = cv2.GaussianBlur(img, (0, 0), sigma)
        out = cv2.addWeighted(img, 1.0 + amount, blurred, -amount, 0)
        return np.clip(out, 0, 255).astype(np.uint8)

    # ── Percentile stretch ──
    @staticmethod
    def _percentile_stretch(
        img: np.ndarray, lo: float = 1.0, hi: float = 99.0
    ) -> np.ndarray:
        out = np.zeros_like(img)
        for c in range(3):
            ch = img[:, :, c].astype(float)
            p_lo, p_hi = np.percentile(ch, [lo, hi])
            ch = np.clip((ch - p_lo) / max(p_hi - p_lo, 1e-6), 0, 1) * 255.0
            out[:, :, c] = ch.astype(np.uint8)
        return out

    # ── Grey-world white balance ──
    @staticmethod
    def _grey_world_white_balance(img: np.ndarray) -> np.ndarray:
        """Assumes the average colour of the scene is neutral grey."""
        img_f = img.astype(np.float32)
        mean_b = np.mean(img_f[:, :, 0])
        mean_g = np.mean(img_f[:, :, 1])
        mean_r = np.mean(img_f[:, :, 2])
        gray_mean = (mean_b + mean_g + mean_r) / 3.0
        scale_b = gray_mean / max(mean_b, 1.0)
        scale_g = gray_mean / max(mean_g, 1.0)
        scale_r = gray_mean / max(mean_r, 1.0)
        img_f[:, :, 0] *= scale_b
        img_f[:, :, 1] *= scale_g
        img_f[:, :, 2] *= scale_r
        return np.clip(img_f, 0, 255).astype(np.uint8)

    # ── ACES tone mapping (HDR→SDR) ──
    @staticmethod
    def _aces_tone_map(img: np.ndarray) -> np.ndarray:
        """
        Approximation of ACES filmic tone mapping curve.
        Compresses highlights while preserving mid-tones.
        """
        x = img.astype(np.float32) / 255.0
        a, b, c, d, e = 2.51, 0.03, 2.43, 0.59, 0.14
        mapped = (x * (a * x + b)) / (x * (c * x + d) + e)
        mapped = np.clip(mapped, 0.0, 1.0)
        return (mapped * 255.0).astype(np.uint8)

    # ── Saturation boost ──
    @staticmethod
    def _boost_saturation(img: np.ndarray, scale: float = 1.30) -> np.ndarray:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * scale, 0, 255)
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
