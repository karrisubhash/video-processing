"""
DVPE — Drone Video Preprocessing Engine orchestrator (maximal-retention model).

Flow:

    Upload/session -> Validation -> Metadata -> Decode Every Frame ->
    Quality Analysis (reject ONLY unusable) -> Near-Duplicate Removal (only near-identical) ->
    Coverage Check (safety rule: no gap > max_gap_seconds) -> Best Frame Selection -> Output

Preprocessing ONLY: no detection, tracking, YOLO, SAM, or enhancement. The engine keeps
every genuinely useful, distinct frame and only removes what is clearly worthless or
redundant, so the output scales with the video's real content. Several signals are used
together (blur + brightness + contrast for quality; perceptual-hash + SSIM for duplicates;
a time-interval safety rule for coverage) — never a single feature.

Produces:
    data/missions/<mission_id>/
        best_frames/            the selected frames, copied unmodified
        metadata.json           video metadata
        processing_summary.json validation metrics + ranked best-frame manifest

Run:
    python -m video_processing.dvpe.pipeline --video path/to/video.mp4 --mission-id Mission_001
"""
from __future__ import annotations
import argparse
import shutil
import sys
import time
from pathlib import Path

from video_processing.dvpe.config import DVPEConfig, DEFAULT_DVPE_CONFIG
from video_processing.dvpe import extraction, quality_gate, dedup as dvpe_dedup, coverage as coverage_mod, scoring, selection, output
from video_processing.dvpe.metadata import extract_video_metadata
from video_processing.services import validator
from video_processing.utils.io_helpers import get_logger

logger = get_logger("dvpe.pipeline")


def run_dvpe(video_path: str, mission_id: str, config: DVPEConfig = None) -> dict:
    config = config or DEFAULT_DVPE_CONFIG
    t0 = time.time()
    timings: dict = {}
    mission_root = config.data_root / "missions" / mission_id
    frames_dir = mission_root / "frames"
    best_dir = mission_root / "best_frames"

    logger.info(f"=== DVPE preprocessing starting: {mission_id} ===")

    # Validation
    ts = time.time()
    result = validator.validate_video(video_path, config.validation)
    timings["validation"] = round(time.time() - ts, 2)
    if not result.accepted:
        logger.error(f"Video rejected: {result.reasons}")
        return {"mission_id": mission_id, "accepted": False, "reasons": result.reasons}

    # Metadata
    ts = time.time()
    video_metadata = extract_video_metadata(video_path, mission_id)
    timings["metadata"] = round(time.time() - ts, 2)
    fps = video_metadata.get("fps") or 30.0

    # Decode every frame
    ts = time.time()
    records = extraction.decode_all_frames(video_path, str(frames_dir), config.decode)
    timings["decode"] = round(time.time() - ts, 2)
    if not records:
        return _finish_empty(config, mission_root, best_dir, video_metadata, mission_id, timings, t0)

    # Quality analysis (metrics already computed inline during decode; reject only unusable)
    ts = time.time()
    _, qinfo = quality_gate.apply_quality(records, config.quality_reject)
    timings["quality"] = round(time.time() - ts, 2)
    n_rejected_quality = len(records) - qinfo["passed"]

    # Near-duplicate removal (only among usable frames; keep the sharpest of each cluster).
    # Inspect mode compares against a history of recently-kept frames, not just the last one.
    ts = time.time()
    usable = [r for r in records if r.passed_quality]
    dedup_history = config.inspect.history_size if config.selection.mode == "inspect" else config.dedup_history_size
    _, n_duplicates = dvpe_dedup.remove_near_duplicates(
        usable, config.duplicate, prefer_sharpest=True, history_size=dedup_history)
    timings["duplicate"] = round(time.time() - ts, 2)

    # Frame scoring (image quality only) — used by selection/coverage to pick representatives
    ts = time.time()
    scoring.score_frames([r for r in records if r.passed_quality and not r.is_duplicate],
                         config.scoring, config.quality)
    timings["scoring"] = round(time.time() - ts, 2)

    # Best frame selection — density-capped (auto) so the output is proportionate to duration,
    # or composite + motion-adaptive (inspect).
    ts = time.time()
    selected = selection.select_best_frames(records, config.selection, fps, inspect_cfg=config.inspect)
    timings["selection"] = round(time.time() - ts, 2)

    # Coverage check (safety rule) — force-keep frames so no gap in the KEPT set exceeds the limit
    ts = time.time()
    selected, forced = coverage_mod.fill_gaps(selected, records, fps, config.coverage)
    timings["coverage"] = round(time.time() - ts, 2)
    # Finalize the selected flag (fill_gaps may have added frames).
    selected_ids = {r.frame_id for r in selected}
    for r in records:
        r.selected_as_best = r.frame_id in selected_ids

    # Output: copy selected frames unmodified into best_frames/
    ts = time.time()
    best_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    for record in selected:
        src = Path(record.path)
        if src.exists():
            shutil.copy2(src, best_dir / f"{record.frame_id}.jpg")
            copied += 1
    timings["output"] = round(time.time() - ts, 2)
    logger.info(f"Output: copied {copied} best frame(s) -> {best_dir}")

    largest_gap = coverage_mod.largest_gap_frames(selected)
    elapsed = round(time.time() - t0, 2)
    summary = {
        "mission_id": mission_id,
        "accepted": True,
        "video_name": video_metadata.get("video_name"),
        # ---- validation metrics ----
        "original_frame_count": video_metadata.get("frame_count"),
        "frames_decoded": len(records),
        "frames_rejected_by_quality": n_rejected_quality,
        "quality_rejections": qinfo["rejected"],
        "frames_removed_as_duplicates": n_duplicates,
        "coverage_forced_frames": forced,
        "frames_kept": len(selected),
        "largest_gap_frames": largest_gap,
        "largest_gap_seconds": round(largest_gap / fps, 3) if fps else None,
        "selection_mode": config.selection.mode,
        "timings_sec": timings,
        "processing_time_sec": elapsed,
    }

    output.write_outputs(mission_root, video_metadata, selected, summary, best_dir)

    if not config.keep_extracted_frames:
        shutil.rmtree(frames_dir, ignore_errors=True)
        logger.info("Removed intermediate frames cache (set keep_extracted_frames=True to retain)")

    logger.info(f"=== DVPE complete in {elapsed}s: kept {len(selected)} frame(s), "
                f"largest gap {summary['largest_gap_seconds']}s -> {best_dir} ===")
    return summary


def _finish_empty(config, mission_root, best_dir, video_metadata, mission_id, timings, t0) -> dict:
    logger.warning("No frames decoded — emitting an empty package.")
    summary = {"mission_id": mission_id, "accepted": True, "frames_kept": 0,
               "timings_sec": timings, "processing_time_sec": round(time.time() - t0, 2)}
    best_dir.mkdir(parents=True, exist_ok=True)
    output.write_outputs(mission_root, video_metadata, [], summary, best_dir)
    if not config.keep_extracted_frames:
        shutil.rmtree(mission_root / "frames", ignore_errors=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description="Run the DVPE drone-video preprocessing engine.")
    parser.add_argument("--video", required=True, help="Path to input drone video")
    parser.add_argument("--mission-id", required=True, help="Mission/session id, e.g. Mission_001")
    parser.add_argument("--config", default=None,
                        help="Optional JSON config file with weight/threshold overrides")
    parser.add_argument("--mode", choices=["auto", "all", "count", "fraction", "inspect"], default=None,
                        help="Selection mode (default auto = keep up to --fps frames/sec; "
                             "inspect = composite quality+uniqueness+coverage+diversity, motion-adaptive)")
    parser.add_argument("--fps", type=float, default=None,
                        help="auto mode: max kept frames per second of video (default 2.0)")
    parser.add_argument("--target-count", type=int, default=None, help="For --mode count: number of frames")
    parser.add_argument("--target-fraction", type=float, default=None, help="For --mode fraction: fraction to keep")
    parser.add_argument("--max-gap-seconds", type=float, default=None,
                        help="Coverage safety rule: max seconds between kept frames (default 1.0)")
    parser.add_argument("--max-output", type=int, default=None, help="Soft cap on number of output frames")
    parser.add_argument("--keep-frames", action="store_true",
                        help="Keep the intermediate decoded-frames cache (default: removed)")
    args = parser.parse_args()

    # Config file first (if any); CLI flags below override it.
    config = DVPEConfig()
    if args.config is not None:
        from video_processing.dvpe.config_io import load_config
        config = load_config(args.config, base=config)
    if args.mode is not None:
        config.selection.mode = args.mode
    if args.fps is not None:
        config.selection.keep_fps = args.fps
    if args.target_count is not None:
        config.selection.target_count = args.target_count
        if args.mode is None:
            config.selection.mode = "count"
    if args.target_fraction is not None:
        config.selection.target_fraction = args.target_fraction
        if args.mode is None:
            config.selection.mode = "fraction"
    if args.max_gap_seconds is not None:
        config.coverage.max_gap_seconds = args.max_gap_seconds
    if args.max_output is not None:
        config.selection.max_output_frames = args.max_output
    if args.keep_frames:
        config.keep_extracted_frames = True

    result = run_dvpe(args.video, args.mission_id, config)
    if not result.get("accepted"):
        sys.exit(1)


if __name__ == "__main__":
    main()
