"""
DVPE — APTRANSCO Drone Video Preprocessing Engine.

A preprocessing-only pipeline: it turns a raw drone video into a small set of the
highest-quality, non-duplicate, enhanced inspection frames plus metadata and a
processing summary. It deliberately contains NO tower detection, tracking, YOLO,
SAM, or inspection logic — that is left to whatever inspection system consumes the
`best_frames/` output.

Reuses the shared services in ``video_processing.services`` for the stages it has in
common with the full DVIE engine; the preprocessing-specific scoring, selection, and
flat output live here.

Entry point: ``from video_processing.dvpe.pipeline import run_dvpe``
(not re-exported here, so ``python -m video_processing.dvpe.pipeline`` doesn't warn
about a double import).
"""

