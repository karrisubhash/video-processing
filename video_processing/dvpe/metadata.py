"""
Module 3 — Video Metadata Extraction (DVPE).

Produces exactly the fields the DVPE spec calls for: video name, duration, fps,
resolution, frame count, codec, file size, and capture time (if available). Reuses the
shared probe (ffprobe when installed, OpenCV otherwise) and adds file size + a
best-effort container capture-time tag.
"""
from __future__ import annotations
import os
import subprocess
from pathlib import Path
from typing import Optional

from video_processing.utils.probe import probe_video, ffprobe_available
from video_processing.utils.io_helpers import get_logger

logger = get_logger("dvpe.metadata")


def _capture_time(video_path: str) -> Optional[str]:
    """Best-effort container capture time from the ffprobe ``creation_time`` tag.
    Returns None when unavailable (we do NOT fall back to file mtime — that would be a
    misleading proxy for when the footage was actually captured)."""
    if ffprobe_available():
        try:
            out = subprocess.run(
                ["ffprobe", "-v", "error", "-show_entries", "format_tags=creation_time",
                 "-of", "default=nw=1:nk=1", str(video_path)],
                capture_output=True, text=True, timeout=15,
            )
            val = (out.stdout or "").strip()
            if val:
                return val
        except Exception as e:
            logger.warning(f"capture_time probe failed: {e}")
    return None


def extract_video_metadata(video_path: str, mission_id: str) -> dict:
    """Module 3 output as a plain dict (written verbatim to metadata.json)."""
    p = Path(video_path)
    probe = probe_video(video_path) or {}
    w, h = probe.get("width"), probe.get("height")
    meta = {
        "mission_id": mission_id,
        "video_name": p.name,
        "duration_sec": probe.get("duration"),
        "fps": probe.get("fps"),
        "resolution": [w, h] if w and h else None,
        "frame_count": probe.get("nb_frames"),
        "codec": probe.get("codec_name"),
        "file_size_bytes": (os.path.getsize(video_path) if p.exists() else None),
        "capture_time": _capture_time(video_path),
        "probe_source": probe.get("source"),
    }
    logger.info(
        f"Module 3: metadata for {mission_id}: {meta['resolution']} @ {meta['fps']}fps, "
        f"{meta['duration_sec']}s, {meta['codec']}, "
        f"{(meta['file_size_bytes'] or 0) / 1e6:.1f} MB"
    )
    return meta
