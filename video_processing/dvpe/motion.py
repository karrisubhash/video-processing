"""
Camera-motion estimation (DVPE inspect mode).

The preprocessing engine decodes every frame and stashes a small grayscale thumbnail
(``_gray_small``) on each record. This module turns those into a per-frame *motion*
signal — the mean absolute pixel difference to the previous decoded frame — which is a
cheap proxy for how much the camera/scene moved between frames. No new dependency and no
disk re-reads (it reuses the in-memory thumbnails from decode).

The signal drives two inspect-mode improvements:
  - Adaptive selection window (sample dynamic segments more densely than static ones).
  - Viewpoint-diversity scoring (a frame far in cumulative motion from the last kept one
    likely shows a new angle).

A short moving-average smooths single-frame spikes (compression noise, a bird crossing).
"""
from __future__ import annotations

import cv2

from video_processing.utils.image_ops import mean_abs_diff
from video_processing.utils.io_helpers import get_logger

logger = get_logger("dvpe.motion")


def compute_motion(records: list, smooth: int = 3) -> list:
    """Sets ``record.motion`` (mean-abs-diff to the previous decoded frame) on each record,
    in place, in timeline order. Idempotent-friendly: safe to call more than once. Returns
    the same list. Frames without a stashed thumbnail get motion 0.0 (treated as static)."""
    ordered = sorted(records, key=lambda r: r.frame_index)
    prev_gray = None
    raw: list = []
    for r in ordered:
        gray = getattr(r, "_gray_small", None)
        if gray is None or prev_gray is None:
            m = 0.0
        else:
            a, b = prev_gray, gray
            if a.shape != b.shape:
                b = cv2.resize(b, (a.shape[1], a.shape[0]))
            m = mean_abs_diff(a, b)
        r.motion = round(m, 3)
        raw.append(r)
        if gray is not None:
            prev_gray = gray

    if smooth and smooth > 1 and len(raw) > 2:
        vals = [r.motion or 0.0 for r in raw]
        half = smooth // 2
        for i, r in enumerate(raw):
            lo, hi = max(0, i - half), min(len(vals), i + half + 1)
            r.motion = round(sum(vals[lo:hi]) / (hi - lo), 3)

    if raw:
        peak = max(r.motion or 0.0 for r in raw)
        logger.info(f"Motion estimated for {len(raw)} frame(s) (peak mean-abs-diff {peak})")
    return records
