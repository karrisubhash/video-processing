"""Low-level image metric helpers shared across Layer 4 (quality), Layer 9 (scoring),
and Layer 11 (enhancement). Kept dependency-light (cv2 + numpy only).
"""
from __future__ import annotations
import cv2
import numpy as np


def variance_of_laplacian(gray: np.ndarray) -> float:
    """Sharpness metric. Higher = sharper. Standard blur-detection measure."""
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def mean_brightness(gray: np.ndarray) -> float:
    return float(np.mean(gray))


def rms_contrast(gray: np.ndarray) -> float:
    return float(np.std(gray))


def estimate_noise_sigma(gray: np.ndarray) -> float:
    """Fast noise estimate via Laplacian-based method (Immerkaer, 1996)."""
    h, w = gray.shape
    if h < 3 or w < 3:
        return 0.0
    kernel = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float64)
    conv = cv2.filter2D(gray.astype(np.float64), -1, kernel)
    sigma = np.sum(np.abs(conv)) * np.sqrt(0.5 * np.pi) / (6 * (w - 2) * (h - 2))
    return float(sigma)


def exposure_flags(gray: np.ndarray, low: float = 15, high: float = 240) -> dict:
    """Fraction of pixels clipped dark/bright, used to flag over/under-exposure."""
    total = gray.size
    dark_frac = float(np.sum(gray <= low)) / total
    bright_frac = float(np.sum(gray >= high)) / total
    return {
        "underexposed": dark_frac > 0.35,
        "overexposed": bright_frac > 0.35,
        "dark_fraction": dark_frac,
        "bright_fraction": bright_frac,
    }


def to_gray(frame_bgr: np.ndarray) -> np.ndarray:
    if frame_bgr.ndim == 2:
        return frame_bgr
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)


def hist_correlation(gray_a: np.ndarray, gray_b: np.ndarray) -> float:
    """Correlation between grayscale histograms, in [-1, 1]. Used for scene-change detection."""
    hist_a = cv2.calcHist([gray_a], [0], None, [64], [0, 256])
    hist_b = cv2.calcHist([gray_b], [0], None, [64], [0, 256])
    cv2.normalize(hist_a, hist_a)
    cv2.normalize(hist_b, hist_b)
    return float(cv2.compareHist(hist_a, hist_b, cv2.HISTCMP_CORREL))


def mean_abs_diff(gray_a: np.ndarray, gray_b: np.ndarray) -> float:
    """Simple motion proxy between consecutive frames."""
    return float(np.mean(cv2.absdiff(gray_a, gray_b)))
