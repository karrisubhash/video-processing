"""
Tower-grouping orchestrator (spec Steps 6-7 + the complete workflow).

Consumes a finished DVPE mission (``best_frames/`` + ``processing_summary.json``), resolves
each frame's GPS from the supplied track, assigns it to a tower with the hysteresis rule, then
lays out the tower folders and metadata:

    Mission_XX/
        best_frames/                 (left intact in copy mode)
        Tower_001/
            frameNNNNN.jpg ...
            metadata.json            {tower_id, frames, start_time, end_time, best_frame, ...}
        Tower_002/ ...
        tower_grouping_summary.json  overall audit trail
"""
from __future__ import annotations
import re
import shutil
from pathlib import Path
from typing import Optional

from video_processing.tower_grouping.assign import FrameGeo, assign_frames
from video_processing.tower_grouping.config import TowerGroupingConfig, DEFAULT_TOWER_GROUPING_CONFIG
from video_processing.tower_grouping.gps import load_gps_track
from video_processing.tower_grouping.towers import load_tower_db
from video_processing.utils.io_helpers import get_logger, read_json, write_json

logger = get_logger("tower_grouping")

_TRAILING_NUM = re.compile(r"(\d+)\s*$")


def _resolve_mission_dir(mission_id: Optional[str], mission_dir: Optional[str],
                         data_root: Optional[str]) -> Path:
    if mission_dir:
        return Path(mission_dir)
    if not mission_id:
        raise ValueError("Provide either mission_dir or mission_id.")
    if data_root is None:
        from video_processing.config import _PACKAGE_ROOT
        data_root = _PACKAGE_ROOT / "data"
    return Path(data_root) / "missions" / mission_id


def _load_frames(mission_dir: Path) -> list:
    """Read the DVPE best-frame manifest into a list of (frame_id, timestamp, file, score),
    in timeline order."""
    summary_path = mission_dir / "processing_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(
            f"No processing_summary.json in {mission_dir} — run DVPE first.")
    summary = read_json(summary_path)
    entries = summary.get("best_frames") or []
    frames = []
    for e in entries:
        frames.append({
            "frame_id": e.get("frame_id"),
            "timestamp_sec": e.get("timestamp_sec") or 0.0,
            "file": e.get("file") or f"best_frames/{e.get('frame_id')}.jpg",
            "score": e.get("score"),
            "frame_index": e.get("frame_index", 0),
        })
    frames.sort(key=lambda f: (f["frame_index"], f["timestamp_sec"]))
    return frames


def _folder_name(tower_id: str, config: TowerGroupingConfig, used: set, seq: int) -> str:
    m = _TRAILING_NUM.search(tower_id)
    num = int(m.group(1)) if m else seq
    base = f"{config.folder_prefix}{num:0{config.folder_pad}d}"
    name, k = base, 1
    while name in used:                      # guarantee uniqueness across towers
        name, k = f"{base}_{k}", k + 1
    used.add(name)
    return name


def run_tower_grouping(towers_path: str, gps_path: str, *,
                       mission_id: str = None, mission_dir: str = None,
                       data_root: str = None,
                       config: TowerGroupingConfig = None) -> dict:
    """Group a mission's best frames tower-wise. Returns the grouping summary dict."""
    config = config or DEFAULT_TOWER_GROUPING_CONFIG
    mdir = _resolve_mission_dir(mission_id, mission_dir, data_root)
    if not mdir.exists():
        raise FileNotFoundError(f"Mission directory not found: {mdir}")
    logger.info(f"=== Tower grouping starting: {mdir.name} ===")

    towers = load_tower_db(towers_path)
    track = load_gps_track(gps_path)
    tower_by_id = {t.tower_id: t for t in towers}

    # Build FrameGeo list: resolve each frame's GPS from the track.
    raw = _load_frames(mdir)
    geos, gps_hits = [], 0
    for f in raw:
        sample = track.lookup(f["timestamp_sec"], frame_id=f["frame_id"],
                              interpolate=config.interpolate_gps,
                              max_time_gap_sec=config.gps_max_time_gap_sec)
        if sample is not None:
            gps_hits += 1
        geos.append(FrameGeo(
            frame_id=f["frame_id"], timestamp_sec=f["timestamp_sec"], file=f["file"],
            score=f["score"],
            lat=sample.lat if sample else None,
            lon=sample.lon if sample else None,
        ))
    logger.info(f"Resolved GPS for {gps_hits}/{len(geos)} frame(s)")

    result = assign_frames(geos, towers, config)

    # Group assignments by tower, preserving first-appearance order for folder numbering.
    order, by_tower = [], {}
    for a in result.assignments:
        if a.tower_id is None:
            continue
        if a.tower_id not in by_tower:
            by_tower[a.tower_id] = []
            order.append(a.tower_id)
        by_tower[a.tower_id].append(a)

    used_names, tower_records, seq = set(), [], 1
    for tid in order:
        assigned = by_tower[tid]
        folder = _folder_name(tid, config, used_names, seq)
        seq += 1
        tdir = mdir / folder
        tdir.mkdir(parents=True, exist_ok=True)

        written, dists = 0, []
        for a in assigned:
            src = mdir / a.file
            dst = tdir / Path(a.file).name
            if src.exists():
                (shutil.move if config.move_frames else shutil.copy2)(str(src), str(dst))
                written += 1
            else:
                logger.warning(f"Source frame missing, skipped: {src}")
            if a.distance_m is not None:
                dists.append(a.distance_m)

        best = max(assigned, key=lambda a: (a.score if a.score is not None else -1.0))
        t = tower_by_id[tid]
        meta = {
            "tower_id": tid,
            "tower_lat": t.lat,
            "tower_lon": t.lon,
            "frames": len(assigned),
            "start_time": round(min(a.timestamp_sec for a in assigned), 3),
            "end_time": round(max(a.timestamp_sec for a in assigned), 3),
            "best_frame": f"{best.frame_id}.jpg",
            "mean_distance_m": round(sum(dists) / len(dists), 2) if dists else None,
            "max_distance_m": round(max(dists), 2) if dists else None,
            "frame_files": [f"{a.frame_id}.jpg" for a in assigned],
        }
        write_json(tdir / "metadata.json", meta)
        tower_records.append({
            "tower_id": tid, "folder": folder, "frames": len(assigned),
            "start_time": meta["start_time"], "end_time": meta["end_time"],
            "best_frame": meta["best_frame"], "frames_written": written,
        })
        logger.info(f"{folder}: {len(assigned)} frame(s) [{meta['start_time']}s-{meta['end_time']}s], "
                    f"best={meta['best_frame']}")

    reasons = {}
    for a in result.assignments:
        if a.tower_id is None:
            reasons[a.unassigned_reason] = reasons.get(a.unassigned_reason, 0) + 1

    summary = {
        "mission_id": mdir.name,
        "gps_source": {"path": str(track.source_path), "kind": track.source_kind, "fixes": len(track)},
        "tower_db": {"path": str(towers_path), "towers": len(towers)},
        "config": {
            "switch_patience": config.switch_patience,
            "max_assign_distance_m": config.max_assign_distance_m,
            "interpolate_gps": config.interpolate_gps,
            "gps_max_time_gap_sec": config.gps_max_time_gap_sec,
            "move_frames": config.move_frames,
        },
        "frames_total": len(geos),
        "frames_with_gps": gps_hits,
        "frames_assigned": len(geos) - result.unassigned,
        "frames_unassigned": result.unassigned,
        "unassigned_reasons": reasons,
        "tower_switches": result.switches,
        "towers_populated": len(tower_records),
        "towers": tower_records,
    }
    write_json(mdir / "tower_grouping_summary.json", summary)
    logger.info(f"=== Tower grouping complete: {len(tower_records)} tower folder(s), "
                f"{summary['frames_assigned']}/{summary['frames_total']} frame(s) assigned "
                f"-> {mdir} ===")
    return summary
