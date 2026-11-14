"""
Quality Analysis (DVPE) — reject ONLY unusable frames.

Deliberately lenient and multi-signal (never depends on one feature). A frame is rejected
ONLY when we're confident it adds no value:
  - corrupted / unreadable
  - completely black or completely white (mean brightness or clipped-pixel fraction extreme)
  - poor visibility (contrast/std near zero)
  - extremely blurred (noise-normalized variance-of-Laplacian below a low floor)

Everything else remains a candidate. Sharpness is noise-normalized so high-ISO grain (which
inflates raw Laplacian variance) is not mistaken for detail. Returns (records, info).
"""
from __future__ import annotations

from video_processing.dvpe.config import QualityRejectConfig, DEFAULT_DVPE_CONFIG
from video_processing.utils.io_helpers import get_logger

logger = get_logger("dvpe.quality")


def _effective_sharpness(r, k: float) -> float:
    return (r.sharpness or 0.0) / (1.0 + k * (r.noise_sigma or 0.0))


def _rejection_reason(r, cfg: QualityRejectConfig):
    if r.sharpness is None or r.brightness is None:
        return "corrupted"
    if r.brightness <= cfg.black_brightness or (r.dark_fraction or 0) > cfg.dark_fraction_max:
        return "black"
    if r.brightness >= cfg.white_brightness or (r.bright_fraction or 0) > cfg.bright_fraction_max:
        return "white"
    if (r.contrast or 0) < cfg.min_contrast:
        return "no_contrast"
    if _effective_sharpness(r, cfg.noise_penalty_k) < cfg.extreme_blur_floor:
        return "extreme_blur"
    return None


def apply_quality(records: list, cfg: QualityRejectConfig = None):
    """Sets passed_quality on each record (True unless clearly unusable). Records the
    rejection reason in keep_reasons for transparency. Returns (records, info)."""
    cfg = cfg or DEFAULT_DVPE_CONFIG.quality_reject
    reasons: dict = {}
    passed = 0
    for r in records:
        reason = _rejection_reason(r, cfg)
        r.passed_quality = reason is None
        if reason is None:
            passed += 1
        else:
            r.keep_reasons = list(r.keep_reasons or []) + [f"rejected:{reason}"]
            reasons[reason] = reasons.get(reason, 0) + 1
    logger.info(f"Quality analysis: {passed}/{len(records)} usable (rejected={reasons or 'none'})")
    return records, {"passed": passed, "total": len(records), "rejected": reasons}
