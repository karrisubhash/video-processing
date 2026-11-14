"""
Module 8 — Frame Scoring (DVPE).

Ranks the frames that survived quality filtering + duplicate removal on IMAGE QUALITY
ONLY — sharpness, contrast, brightness, noise, exposure. No detection/tower/geometry
signals (that is the whole point of the preprocessing-only engine).

Reuses the per-frame metrics already computed in Module 5 (quality assessment), so this
stage does NOT re-read images from disk — it just combines the stored metrics into one
0-100 ``final_score`` per frame, which Module 9 selects on.
"""
from __future__ import annotations

from video_processing.config import QualityConfig, DEFAULT_CONFIG
from video_processing.dvpe.config import DVPEScoringConfig, DEFAULT_DVPE_CONFIG
from video_processing.models.schema import FrameRecord
from video_processing.utils.io_helpers import get_logger

logger = get_logger("dvpe.scoring")


def _normalize(value: float, good: float, bad: float) -> float:
    """Linearly map value to 0-100. ``good`` -> 100, ``bad`` (or worse) -> 0."""
    if good == bad:
        return 100.0
    frac = (value - bad) / (good - bad)
    return max(0.0, min(100.0, frac * 100.0))


def score_frame(record: FrameRecord, scfg: DVPEScoringConfig, qcfg: QualityConfig) -> FrameRecord:
    # Module 5 must have run first; if a frame has no metrics it scores 0.
    if record.sharpness is None:
        record.final_score = 0.0
        record.score_breakdown = {}
        return record

    sharpness = _normalize(record.sharpness, good=qcfg.blur_threshold * 4, bad=qcfg.blur_threshold)
    mid_bright = (qcfg.min_brightness + qcfg.max_brightness) / 2
    half_range = (qcfg.max_brightness - qcfg.min_brightness) / 2 or 1.0
    brightness = max(0.0, 100.0 - abs((record.brightness or 0) - mid_bright) / half_range * 100.0)
    contrast = _normalize(record.contrast or 0, good=qcfg.min_contrast * 3, bad=qcfg.min_contrast * 0.5)
    noise = _normalize(record.noise_sigma or 0, good=qcfg.max_noise_sigma * 0.2, bad=qcfg.max_noise_sigma * 1.5)
    exposure = 0.0 if (record.overexposed or record.underexposed) else 100.0

    breakdown = {
        "sharpness": sharpness,
        "contrast": contrast,
        "brightness": brightness,
        "noise": noise,
        "exposure": exposure,
    }
    weights = scfg.weights
    final = sum(breakdown[k] * weights.get(k, 0.0) for k in breakdown)

    record.score_breakdown = {k: round(v, 2) for k, v in breakdown.items()}
    record.final_score = round(final, 2)
    return record


def score_frames(records: list, scfg: DVPEScoringConfig = None, qcfg: QualityConfig = None) -> list:
    scfg = scfg or DEFAULT_DVPE_CONFIG.scoring
    qcfg = qcfg or DEFAULT_CONFIG.quality
    for r in records:
        score_frame(r, scfg, qcfg)
    if records:
        top = max(r.final_score or 0 for r in records)
        logger.info(f"Module 8: scored {len(records)} frame(s) on image quality (top score {top})")
    return records
