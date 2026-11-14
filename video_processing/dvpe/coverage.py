"""
Coverage Check (DVPE) — the temporal safety rule.

"Never allow more than max_gap_seconds (e.g., 1s / 30 frames at 30fps) without keeping at
least one frame." Runs AFTER best-frame selection, so it guarantees a lower bound on
temporal density even where selection/dedup/quality left a gap (e.g., a long static stretch
that collapsed to a single frame, or a run of unusable frames).

Walks all decoded frames in timeline order; whenever the gap since the last kept frame
would exceed the limit, it force-keeps the best available frame in that window (preferring
usable, non-duplicate frames; otherwise the least-bad). Forced frames are tagged
``coverage`` in ``keep_reasons``.
"""
from __future__ import annotations

from video_processing.dvpe.config import CoverageConfig, DEFAULT_DVPE_CONFIG
from video_processing.utils.io_helpers import get_logger

logger = get_logger("dvpe.coverage")


def _score(r) -> float:
    if r.final_score is not None:
        return r.final_score
    return (r.sharpness or 0.0) - (100.0 if (r.overexposed or r.underexposed) else 0.0)


def fill_gaps(selected: list, all_records: list, fps: float, config: CoverageConfig = None):
    """Ensures no gap between kept frames exceeds the limit. Mutates/returns
    (augmented_selected, forced_count)."""
    config = config or DEFAULT_DVPE_CONFIG.coverage
    if not all_records:
        return selected, 0
    max_gap = max(1, int(round(config.max_gap_seconds * (fps or 30.0))))

    ordered = sorted(all_records, key=lambda r: r.frame_index)
    selected_ids = {r.frame_id for r in selected}
    forced = 0
    last_kept_idx = None
    window: list = []

    for r in ordered:
        if r.frame_id in selected_ids:
            last_kept_idx = r.frame_index
            window = []
            continue
        window.append(r)
        base = last_kept_idx if last_kept_idx is not None else (ordered[0].frame_index - 1)
        if r.frame_index - base >= max_gap:
            usable = [w for w in window if w.passed_quality and not w.is_duplicate]
            pick = max(usable or window, key=_score)
            pick.keep_reasons = list(pick.keep_reasons or []) + ["coverage"]
            pick.selected_as_best = True
            selected.append(pick)
            selected_ids.add(pick.frame_id)
            forced += 1
            last_kept_idx = pick.frame_index
            window = []

    selected.sort(key=lambda r: r.frame_index)
    logger.info(f"Coverage check: force-kept {forced} frame(s) to keep gaps <= {max_gap} frames ({config.max_gap_seconds}s)")
    return selected, forced


def largest_gap_frames(kept_records: list) -> int:
    idxs = sorted(r.frame_index for r in kept_records)
    if len(idxs) < 2:
        return 0
    return max(b - a for a, b in zip(idxs, idxs[1:]))
