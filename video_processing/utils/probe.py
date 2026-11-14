"""
Shared video probing.

Prefers ffprobe (accurate container/codec info) when the ffmpeg CLI is installed,
and falls back to OpenCV so the pipeline runs end-to-end even when ffmpeg/ffprobe is
not on PATH (e.g. a stock Windows box). Both paths return the same normalized dict:

    {
        "codec_name": str | None,   # canonicalized: h264 / hevc / mpeg4 / ...
        "width":      int | None,
        "height":     int | None,
        "fps":        float | None,
        "duration":   float | None, # seconds
        "nb_frames":  int | None,
        "source":     "ffprobe" | "opencv",
    }

Returns None only when the file cannot be read by either method.
"""
from __future__ import annotations
import json
import shutil
import subprocess
from pathlib import Path
from typing import Optional

import cv2

from video_processing.utils.io_helpers import get_logger

logger = get_logger("utils.probe")

# OpenCV FOURCC -> canonical codec name, aligned with ffprobe's codec_name where possible.
_FOURCC_MAP = {
    "avc1": "h264", "h264": "h264", "x264": "h264", "davc": "h264",
    "hev1": "hevc", "hvc1": "hevc", "hevc": "hevc", "h265": "hevc", "x265": "hevc",
    "mp4v": "mpeg4", "mpeg4": "mpeg4", "xvid": "mpeg4", "divx": "mpeg4",
    "fmp4": "mpeg4", "dx50": "mpeg4",
    "vp80": "vp8", "vp90": "vp9", "av01": "av1",
    "mjpg": "mjpeg", "jpeg": "mjpeg",
}


def ffprobe_available() -> bool:
    return shutil.which("ffprobe") is not None


def _parse_fps(r_frame_rate: str) -> Optional[float]:
    try:
        num, den = r_frame_rate.split("/")
        den = float(den)
        return round(float(num) / den, 3) if den else None
    except Exception:
        return None


def _probe_ffprobe(path: Path) -> Optional[dict]:
    try:
        out = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-show_entries",
                "format=duration:stream=codec_name,codec_type,width,height,r_frame_rate,avg_frame_rate,nb_frames",
                "-of", "json", str(path),
            ],
            capture_output=True, text=True, timeout=30,
        )
        if out.returncode != 0:
            logger.warning(f"ffprobe failed: {out.stderr.strip()}")
            return None
        data = json.loads(out.stdout)
        vstream = next(
            (s for s in data.get("streams", []) if s.get("codec_type") == "video"),
            None,
        )
        if vstream is None:
            return None

        duration = None
        try:
            dur_raw = data.get("format", {}).get("duration")
            duration = float(dur_raw) if dur_raw else None
        except (ValueError, TypeError):
            pass

        nb_frames = None
        try:
            nb_frames = int(vstream["nb_frames"]) if vstream.get("nb_frames") else None
        except (ValueError, TypeError):
            pass

        # Prefer avg_frame_rate: r_frame_rate is often the stream TIMEBASE (e.g. 90000/1),
        # not the real frame rate. Fall back to r_frame_rate, then nb_frames/duration.
        avg = _parse_fps(vstream.get("avg_frame_rate", ""))
        r = _parse_fps(vstream.get("r_frame_rate", ""))
        fps = next((c for c in (avg, r) if c and 0 < c <= 1000), None)
        if fps is None and nb_frames and duration:
            fps = round(nb_frames / duration, 3)

        return {
            "codec_name": ((vstream.get("codec_name") or "").lower() or None),
            "width": vstream.get("width"),
            "height": vstream.get("height"),
            "fps": fps,
            "duration": duration,
            "nb_frames": nb_frames,
            "source": "ffprobe",
        }
    except Exception as e:
        logger.warning(f"ffprobe error: {e}")
        return None


def _decode_fourcc(fourcc_value: float) -> str:
    try:
        code = int(fourcc_value)
    except (ValueError, TypeError):
        return ""
    chars = "".join(chr((code >> (8 * i)) & 0xFF) for i in range(4))
    return chars.strip().strip("\x00").lower()


def _probe_opencv(path: Path) -> Optional[dict]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        cap.release()
        return None

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or None
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or None
    fps_raw = cap.get(cv2.CAP_PROP_FPS)
    fps = round(float(fps_raw), 3) if fps_raw and fps_raw > 0 else None
    nb_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or None
    fourcc_raw = _decode_fourcc(cap.get(cv2.CAP_PROP_FOURCC))
    cap.release()

    codec = _FOURCC_MAP.get(fourcc_raw, fourcc_raw or None)
    duration = round(nb_frames / fps, 3) if (nb_frames and fps) else None
    return {
        "codec_name": codec,
        "width": width,
        "height": height,
        "fps": fps,
        "duration": duration,
        "nb_frames": nb_frames,
        "source": "opencv",
    }


def probe_video(video_path: str) -> Optional[dict]:
    """Normalized video probe. Prefers ffprobe, falls back to OpenCV.

    Returns None only if the file can't be read by either method.
    """
    path = Path(video_path)
    if not path.exists():
        return None

    if ffprobe_available():
        result = _probe_ffprobe(path)
        if result is not None:
            return result
        logger.info("ffprobe present but returned no video stream; falling back to OpenCV probe.")

    result = _probe_opencv(path)
    if result is None:
        logger.warning(f"Could not probe video by any method: {video_path}")
    return result
