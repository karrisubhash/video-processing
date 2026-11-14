"""
Best Frame Selection (DVPE).

By the time this runs, quality analysis has rejected only unusable frames and duplicate
removal has dropped only near-identical ones, so the candidate set
(``passed_quality and not is_duplicate``) is the usable, distinct footage. This stage
controls HOW MANY of those to keep so the output is proportionate to the video:

  - ``mode="auto"`` (default): density cap. Keep at most ``keep_fps`` frames per second —
    the highest-scoring frame in each 1/keep_fps-second window. Output scales with duration
    (e.g. 2 fps over 30s => ~60 frames), not with raw frame count.
  - ``mode="all"``: keep every usable, non-duplicate frame (the old maximal-retention).
  - ``mode="count"`` / ``"fraction"``: a bounded top-N by score.
  - ``mode="inspect"``: inspection-aware. "Best frame" = High Quality + Unique Information
    + Good Coverage + Not Duplicate (a composite, not image quality alone), and the sampling
    window is ADAPTIVE to camera motion (hover -> sparse, moving/rotating -> dense). Each kept
    frame records WHY it was kept. See ``_select_inspect``.

The coverage stage runs AFTER this to guarantee a lower bound (no gap > max_gap_seconds).
"""
from __future__ import annotations

from collections import deque

from video_processing.dvpe.config import DVPESelectionConfig, InspectConfig, DEFAULT_DVPE_CONFIG
from video_processing.utils.io_helpers import get_logger

logger = get_logger("dvpe.selection")


def _score(r):
    return r.final_score if r.final_score is not None else (r.sharpness or 0.0)


def _spaced_by_score(pool, limit, min_gap):
    selected, ids, idx = [], set(), []
    for r in sorted(pool, key=_score, reverse=True):
        if len(selected) >= limit:
            break
        if all(abs(r.frame_index - i) >= min_gap for i in idx):
            selected.append(r); ids.add(r.frame_id); idx.append(r.frame_index)
    return selected


def select_best_frames(records: list, config: DVPESelectionConfig = None, fps: float = 30.0,
                       inspect_cfg: InspectConfig = None) -> list:
    config = config or DEFAULT_DVPE_CONFIG.selection
    fps = fps or 30.0

    candidates = [r for r in records if r.passed_quality and not r.is_duplicate]
    if not candidates:
        logger.warning("Best-frame selection: no candidates.")
        return []

    mode = config.mode
    if mode == "inspect":
        selected = _select_inspect(records, candidates, inspect_cfg or DEFAULT_DVPE_CONFIG.inspect, fps)
    elif mode == "all" or (mode == "auto" and (not config.keep_fps or config.keep_fps <= 0)):
        selected = list(candidates)
    elif mode == "count" and config.target_count is not None:
        selected = _spaced_by_score(candidates, max(0, config.target_count), config.min_frame_gap)
    elif mode == "fraction":
        selected = _spaced_by_score(candidates, max(1, round(len(candidates) * config.target_fraction)),
                                    config.min_frame_gap)
    else:  # auto — density cap: best-scoring frame per 1/keep_fps-second window
        bin_size = max(1, round(fps / config.keep_fps))
        bins: dict = {}
        for r in candidates:
            b = r.frame_index // bin_size
            if b not in bins or _score(r) > _score(bins[b]):
                bins[b] = r
        selected = list(bins.values())

    # Guardrail: soft cap for pathological cases.
    if len(selected) > config.max_output_frames:
        selected = _spaced_by_score(selected, config.max_output_frames, config.min_frame_gap)

    for r in records:
        r.selected_as_best = False
    for r in selected:
        r.selected_as_best = True
    selected.sort(key=lambda r: r.frame_index)

    logger.info(f"Best-frame selection ({mode}): {len(selected)} of {len(candidates)} candidate frame(s)")
    return selected


# --------------------------------------------------------------------------------------
# Inspect mode — composite scoring + motion-adaptive window + keep reasons
# --------------------------------------------------------------------------------------

def _local_fps(motion: float, ci: InspectConfig) -> float:
    """Map camera motion to a target sampling rate: hover -> sparse, rotating -> dense."""
    if motion <= ci.hover_motion_max:
        return ci.hover_fps
    if motion <= ci.move_motion_max:
        return ci.move_fps
    return ci.rotate_fps


def _novelty(rec, recent, ci: InspectConfig) -> float:
    """0..1 novelty of ``rec`` vs recently-kept frames: 1 = nothing like it kept lately,
    0 = near-identical to something already kept. Uses pHash Hamming distance."""
    ph = getattr(rec, "_phash", None)
    if ph is None or not recent:
        return 1.0
    dists = [ph - getattr(k, "_phash") for k in recent if getattr(k, "_phash", None) is not None]
    if not dists:
        return 1.0
    cap = ci.novelty_cap or 1.0
    return max(0.0, min(1.0, min(dists) / cap))


def _composite(rec, recent, last_sel, cum_by_id, ci: InspectConfig):
    """Return the composite 0-100 score plus its four 0-100 component sub-scores."""
    q = (rec.final_score or 0.0) / 100.0                          # image quality (scoring stage)
    u = _novelty(rec, recent, ci)                                 # unique information
    if last_sel is None:
        d, gap_s = 1.0, 0.0
    else:
        moved = cum_by_id.get(rec.frame_id, 0.0) - cum_by_id.get(last_sel.frame_id, 0.0)
        d = min(1.0, moved / ci.diversity_motion_cap) if ci.diversity_motion_cap > 0 else 0.0
        gap_s = max(0.0, rec.timestamp_sec - last_sel.timestamp_sec)
    c = min(1.0, gap_s / ci.coverage_gap_cap) if ci.coverage_gap_cap > 0 else 0.0

    w = ci.weights
    tot = sum(w.values()) or 1.0
    comp = (w.get("quality", 0) * q + w.get("uniqueness", 0) * u
            + w.get("diversity", 0) * d + w.get("coverage", 0) * c) / tot
    return {"composite": 100.0 * comp, "quality": 100.0 * q,
            "uniqueness": 100.0 * u, "diversity": 100.0 * d, "coverage": 100.0 * c}


def _apply_result(rec, meta, ci: InspectConfig):
    """Overwrite final_score with the composite, record the breakdown, and derive keep reasons."""
    rec.final_score = round(meta["composite"], 2)
    rec.score_breakdown = {k: round(v, 2) for k, v in meta.items()}

    labels = {"highest_quality": meta["quality"], "unique_viewpoint": meta["uniqueness"],
              "camera_movement": meta["diversity"], "coverage_span": meta["coverage"]}
    dominant = max(labels, key=labels.get)
    reasons = [dominant] + [k for k, v in labels.items()
                            if k != dominant and v >= ci.reason_threshold * 100.0]
    rec.keep_reasons = list(dict.fromkeys((rec.keep_reasons or []) + reasons))


def _select_inspect(records: list, candidates: list, ci: InspectConfig, fps: float) -> list:
    """Greedy, motion-adaptive selection. Walks the usable/distinct candidates in timeline
    order, opening a window whose width is 1 / (max local target-fps within it) — so dynamic
    stretches produce shorter windows (more frames) than static ones — and keeps the
    highest-COMPOSITE frame in each window. Uniqueness is measured against the last
    ``history_size`` kept frames; diversity against cumulative motion since the last kept."""
    from video_processing.dvpe.motion import compute_motion

    if any(getattr(r, "motion", None) is None for r in records):
        compute_motion(records)

    # Cumulative motion over the whole timeline, for the diversity (viewpoint-change) term.
    cum, cum_by_id = 0.0, {}
    for r in sorted(records, key=lambda r: r.frame_index):
        cum += (r.motion or 0.0)
        cum_by_id[r.frame_id] = cum

    candidates = sorted(candidates, key=lambda r: r.frame_index)
    selected: list = []
    recent: deque = deque(maxlen=max(1, ci.history_size))
    last_sel = None

    i, n = 0, len(candidates)
    while i < n:
        window = [candidates[i]]
        j = i + 1
        while j < n:
            target_fps = max(_local_fps(w.motion or 0.0, ci) for w in window)
            span = candidates[j].timestamp_sec - window[0].timestamp_sec
            if target_fps > 0 and span >= 1.0 / target_fps:
                break
            window.append(candidates[j])
            j += 1

        best, best_meta = None, None
        for f in window:
            meta = _composite(f, recent, last_sel, cum_by_id, ci)
            if best is None or meta["composite"] > best_meta["composite"]:
                best, best_meta = f, meta
        _apply_result(best, best_meta, ci)
        selected.append(best)
        recent.append(best)
        last_sel = best
        i = j

    logger.info(f"Inspect selection: {len(selected)} frame(s) via composite score + "
                f"motion-adaptive window (hover {ci.hover_fps}/move {ci.move_fps}/rotate {ci.rotate_fps} fps)")
    return selected
