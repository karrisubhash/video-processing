"""
Near-Duplicate Removal (DVPE) — in-memory, fast.

pHash prefilter + SSIM confirmation, keeping the sharpest of a near-identical cluster. Uses
the pHash and small grayscale thumbnail that ``decode`` already stashed on each record — so
it reads NOTHING from disk. Removes only genuinely near-identical frames; distinct views are
preserved.

``history_size`` controls how many recently-kept frames each new frame is compared against:
  - ``1`` (default): compare only against the immediately-previous kept frame (fast, but a
    repeated viewpoint separated by a few frames can slip through).
  - ``N`` (e.g. 5-10, used by inspect mode): compare against the last N kept frames, so
    viewpoints that recur after a short excursion are also caught. Still O(N) per frame.
"""
from __future__ import annotations

from collections import deque

import cv2

from video_processing.config import DuplicateConfig, DEFAULT_CONFIG
from video_processing.utils.io_helpers import get_logger

logger = get_logger("dvpe.dedup")


def remove_near_duplicates(records: list, config: DuplicateConfig = None,
                           prefer_sharpest: bool = True, history_size: int = 1):
    """Marks near-duplicates in place. Compares each frame against the last ``history_size``
    kept frames. Returns (records, n_removed)."""
    config = config or DEFAULT_CONFIG.duplicate
    from skimage.metrics import structural_similarity as ssim

    history: deque = deque(maxlen=max(1, history_size))   # recently-kept records (newest last)
    removed = 0

    def _is_near_dup(a_rec, b_rec) -> bool:
        ah, bh = getattr(a_rec, "_phash", None), getattr(b_rec, "_phash", None)
        if ah is None or bh is None or (ah - bh) > config.phash_hamming_threshold:
            return False
        a, b = a_rec._gray_small, b_rec._gray_small
        if a.shape != b.shape:
            b = cv2.resize(b, (a.shape[1], a.shape[0]))
        try:
            sim = float(ssim(a, b))
        except Exception:
            sim = 1.0
        return sim >= config.ssim_threshold

    for r in records:
        if getattr(r, "_phash", None) is None:
            history.clear()
            history.append(r)
            continue

        match = next((kr for kr in history if _is_near_dup(kr, r)), None)
        if match is not None:
            if prefer_sharpest and (r.sharpness or 0.0) > (match.sharpness or 0.0):
                # The new frame is sharper: keep it, drop the matched kept frame.
                match.is_duplicate = True
                match.duplicate_of = r.frame_id
                r.is_duplicate = False
                try:
                    history.remove(match)
                except ValueError:
                    pass
                history.append(r)
            else:
                r.is_duplicate = True
                r.duplicate_of = match.frame_id
            removed += 1
            continue

        history.append(r)

    logger.info(f"Near-duplicate removal: {removed}/{len(records)} near-identical frame(s) removed "
                f"(history={max(1, history_size)})")
    return records, removed
