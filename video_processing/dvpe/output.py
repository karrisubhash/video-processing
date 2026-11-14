"""
Output Generation (DVPE).

Writes the flat preprocessing package:

    <mission_id>/
        best_frames/            # the selected frames, copied unmodified (NO enhancement)
        metadata.json           # video metadata
        processing_summary.json # counts, timings, and the ranked best-frame manifest

No per-tower folders, no detection artifacts, no enhancement — just the best visual
input for a downstream inspection system. The best-frame images are copied into
``best_dir`` by the pipeline; here we build the manifest that references them.
"""
from __future__ import annotations
from pathlib import Path

from video_processing.utils.io_helpers import get_logger, write_json

logger = get_logger("dvpe.output")


def write_outputs(mission_dir, video_metadata: dict, selected: list, summary: dict, best_dir) -> str:
    """Writes metadata.json + processing_summary.json and returns the mission dir path."""
    mission_dir = Path(mission_dir)
    best_dir = Path(best_dir)

    # Rank by score (1 = highest); scene-change key frames may score low but are still kept.
    by_score = sorted(selected, key=lambda r: (r.final_score or 0.0), reverse=True)
    rank_of = {r.frame_id: i + 1 for i, r in enumerate(by_score)}

    manifest = []
    for r in sorted(selected, key=lambda r: r.frame_index):
        f = best_dir / f"{r.frame_id}.jpg"
        manifest.append({
            "frame_id": r.frame_id,
            "frame_index": r.frame_index,
            "timestamp_sec": r.timestamp_sec,
            "coverage_forced": "coverage" in (r.keep_reasons or []),  # kept by the safety rule
            "reasons": list(r.keep_reasons or []),  # WHY this frame was kept (inspect mode)
            "motion": r.motion,                     # camera-motion proxy (inspect mode)
            "score": r.final_score,
            "score_rank": rank_of.get(r.frame_id),
            "score_breakdown": r.score_breakdown,
            "file": f"best_frames/{r.frame_id}.jpg",
            "written": f.exists(),
        })

    write_json(mission_dir / "metadata.json", video_metadata)

    full_summary = dict(summary)
    full_summary["best_frames"] = manifest
    write_json(mission_dir / "processing_summary.json", full_summary)

    logger.info(
        f"Output: wrote metadata.json + processing_summary.json "
        f"({len(manifest)} best frame(s)) -> {mission_dir}"
    )
    return str(mission_dir)
