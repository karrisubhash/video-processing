"""
Per-frame drone GPS sources (spec Step 2).

DVPE does not capture GPS, so the track is supplied externally. Three formats are
auto-detected by extension:

* ``.srt``  — DJI subtitle sidecar. Both common payload dialects are handled:
              ``[latitude: 17.7348] [longitude: 83.2812]`` and ``GPS(lon,lat,alt)``.
              The cue START time (seconds from video start) is the fix timestamp.
* ``.json`` — a list of fixes, e.g. the spec's
              ``{"frame":"frame00120.jpg","timestamp":5.0,"latitude":..,"longitude":..}``.
              Key aliases are accepted (lat/latitude, lon/lng/longitude, timestamp/time/t).
* ``.csv``  — a header row with the same aliased columns.

A loaded :class:`GpsTrack` resolves a frame's position by exact ``frame_id`` match when the
source provides one, otherwise by timestamp (linear interpolation between the two bracketing
fixes, or nearest-in-time). All timestamps are seconds from the start of the video, the same
clock DVPE stamps onto each frame — so the two line up without extra calibration.
"""
from __future__ import annotations
import bisect
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from video_processing.utils.io_helpers import get_logger

logger = get_logger("tower_grouping.gps")

# Accepted key/column aliases (lower-cased) for the JSON/CSV loaders.
_LAT_KEYS = ("latitude", "lat")
_LON_KEYS = ("longitude", "lon", "lng", "long")
_TIME_KEYS = ("timestamp_sec", "timestamp", "time", "t", "sec", "seconds")
_FRAME_KEYS = ("frame_id", "frame", "file", "filename")
_ALT_KEYS = ("altitude", "alt", "rel_alt", "abs_alt", "height")


@dataclass
class GpsFix:
    lat: float
    lon: float
    timestamp_sec: Optional[float] = None
    alt: Optional[float] = None
    frame_id: Optional[str] = None


@dataclass
class GpsSample:
    """Result of resolving a frame's position: coordinates + how far (in time) the source
    data was from the requested moment (0.0 for an exact frame-id match or in-range
    interpolation)."""
    lat: float
    lon: float
    alt: Optional[float]
    time_gap_sec: float
    source: str            # "frame_id" | "interpolated" | "nearest"


def _norm_frame_id(value) -> Optional[str]:
    """'best_frames/frame00120.jpg' -> 'frame00120'; None stays None."""
    if value is None:
        return None
    stem = Path(str(value)).stem
    return stem or None


def _get(d: dict, keys) -> Optional[object]:
    """First present key (case-insensitive) from a dict, else None."""
    lower = {k.lower(): v for k, v in d.items()}
    for k in keys:
        if k in lower and lower[k] not in (None, ""):
            return lower[k]
    return None


def _to_float(value) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


# --------------------------------------------------------------------------------------
# The track
# --------------------------------------------------------------------------------------
class GpsTrack:
    """A set of GPS fixes with frame-id and timestamp lookup."""

    def __init__(self, fixes: list, source_path: str = "", source_kind: str = ""):
        self.source_path = source_path
        self.source_kind = source_kind
        self.by_frame = {f.frame_id: f for f in fixes if f.frame_id}
        # Time-sorted view for interpolation / nearest-in-time.
        timed = sorted((f for f in fixes if f.timestamp_sec is not None),
                       key=lambda f: f.timestamp_sec)
        self._timed = timed
        self._times = [f.timestamp_sec for f in timed]
        self.count = len(fixes)

    def __len__(self) -> int:
        return self.count

    def lookup(self, timestamp_sec: Optional[float], frame_id: Optional[str] = None,
               interpolate: bool = True, max_time_gap_sec: float = 2.0) -> Optional[GpsSample]:
        """Resolve a position. Exact frame-id match wins; otherwise resolve by time.
        Returns None if no fix is within ``max_time_gap_sec`` of the frame."""
        if frame_id:
            f = self.by_frame.get(_norm_frame_id(frame_id))
            if f is not None:
                return GpsSample(f.lat, f.lon, f.alt, 0.0, "frame_id")

        if timestamp_sec is None or not self._timed:
            return None

        # Locate the insertion point among fix times.
        i = bisect.bisect_left(self._times, timestamp_sec)

        # Exact / within-range interpolation between neighbours i-1 and i.
        if interpolate and 0 < i < len(self._timed):
            a, b = self._timed[i - 1], self._timed[i]
            span = b.timestamp_sec - a.timestamp_sec
            if span <= 0:
                sample = GpsSample(a.lat, a.lon, a.alt, 0.0, "interpolated")
            else:
                w = (timestamp_sec - a.timestamp_sec) / span
                sample = GpsSample(
                    a.lat + (b.lat - a.lat) * w,
                    a.lon + (b.lon - a.lon) * w,
                    None if a.alt is None or b.alt is None else a.alt + (b.alt - a.alt) * w,
                    0.0, "interpolated",
                )
            return sample

        # Outside the track range, or interpolation disabled: nearest-in-time fix.
        candidates = []
        if i < len(self._timed):
            candidates.append(self._timed[i])
        if i > 0:
            candidates.append(self._timed[i - 1])
        nearest = min(candidates, key=lambda f: abs(f.timestamp_sec - timestamp_sec))
        gap = abs(nearest.timestamp_sec - timestamp_sec)
        if gap > max_time_gap_sec:
            return None
        return GpsSample(nearest.lat, nearest.lon, nearest.alt, gap, "nearest")


# --------------------------------------------------------------------------------------
# Loaders
# --------------------------------------------------------------------------------------
def load_gps_track(path: str) -> GpsTrack:
    """Load a GPS track, auto-detecting the format from the file extension."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"GPS source not found: {path}")
    ext = p.suffix.lower()
    if ext == ".srt":
        fixes = _parse_srt(p.read_text(encoding="utf-8", errors="replace"))
        kind = "srt"
    elif ext == ".json":
        fixes = _parse_json(json.loads(p.read_text(encoding="utf-8")))
        kind = "json"
    elif ext == ".csv":
        fixes = _parse_csv(p.read_text(encoding="utf-8"))
        kind = "csv"
    else:
        raise ValueError(f"Unsupported GPS source '{ext}' (expected .srt, .json, or .csv): {path}")
    if not fixes:
        raise ValueError(f"No GPS fixes parsed from {path}")
    logger.info(f"Loaded {len(fixes)} GPS fix(es) from {p.name} ({kind})")
    return GpsTrack(fixes, source_path=str(p), source_kind=kind)


def _fix_from_fields(lat, lon, timestamp=None, frame=None, alt=None) -> Optional[GpsFix]:
    lat_f, lon_f = _to_float(lat), _to_float(lon)
    if lat_f is None or lon_f is None:
        return None
    return GpsFix(lat=lat_f, lon=lon_f, timestamp_sec=_to_float(timestamp),
                  alt=_to_float(alt), frame_id=_norm_frame_id(frame))


def _parse_json(data) -> list:
    if isinstance(data, dict):
        # Accept {"fixes": [...]} / {"track": [...]} / first list value.
        for v in data.values():
            if isinstance(v, list):
                data = v
                break
    if not isinstance(data, list):
        raise ValueError("JSON GPS source must be a list of fixes (or an object containing one).")
    fixes = []
    for item in data:
        if not isinstance(item, dict):
            continue
        fix = _fix_from_fields(
            lat=_get(item, _LAT_KEYS), lon=_get(item, _LON_KEYS),
            timestamp=_get(item, _TIME_KEYS), frame=_get(item, _FRAME_KEYS),
            alt=_get(item, _ALT_KEYS),
        )
        if fix:
            fixes.append(fix)
    return fixes


def _parse_csv(text: str) -> list:
    reader = csv.DictReader(text.splitlines())
    fixes = []
    for row in reader:
        clean = {(k or "").strip(): v for k, v in row.items()}
        fix = _fix_from_fields(
            lat=_get(clean, _LAT_KEYS), lon=_get(clean, _LON_KEYS),
            timestamp=_get(clean, _TIME_KEYS), frame=_get(clean, _FRAME_KEYS),
            alt=_get(clean, _ALT_KEYS),
        )
        if fix:
            fixes.append(fix)
    return fixes


# --- DJI .SRT ---------------------------------------------------------------------------
_TIME_LINE = re.compile(r"(\d{2}):(\d{2}):(\d{2})[,.](\d{1,3})\s*-->")
_LABELLED = re.compile(r"latitude\s*[:=]\s*(-?\d+\.?\d*).*?longitude\s*[:=]\s*(-?\d+\.?\d*)",
                       re.IGNORECASE | re.DOTALL)
_GPS_TUPLE = re.compile(r"GPS\s*\(?\s*(-?\d+\.\d+)\s*[, ]\s*(-?\d+\.\d+)\s*(?:[, ]\s*(-?\d+\.?\d*))?",
                        re.IGNORECASE)
_ALT_LABELLED = re.compile(r"(?:rel_alt|altitude|abs_alt)\s*[:=]\s*(-?\d+\.?\d*)", re.IGNORECASE)


def _cue_start_seconds(time_line: str) -> Optional[float]:
    m = _TIME_LINE.search(time_line)
    if not m:
        return None
    hh, mm, ss, ms = m.groups()
    ms = (ms + "000")[:3]      # normalise to milliseconds
    return int(hh) * 3600 + int(mm) * 60 + int(ss) + int(ms) / 1000.0


def _latlon_from_payload(payload: str) -> Optional[tuple]:
    """Return (lat, lon) from an SRT cue payload, or None. Handles labelled fields and the
    DJI ``GPS(lon,lat,alt)`` tuple; a magnitude check corrects lat/lon if they are swapped."""
    m = _LABELLED.search(payload)
    if m:
        return float(m.group(1)), float(m.group(2))
    m = _GPS_TUPLE.search(payload)
    if m:
        a, b = float(m.group(1)), float(m.group(2))
        lon, lat = a, b                      # DJI documents GPS(longitude, latitude, alt)
        if abs(lat) > 90.0 >= abs(lon):      # clearly swapped -> fix
            lon, lat = lat, lon
        return lat, lon
    return None


def _parse_srt(text: str) -> list:
    """Split into subtitle cues on blank lines; from each cue read the start time and the
    lat/lon (and altitude if present)."""
    fixes = []
    for block in re.split(r"\n\s*\n", text.replace("\r\n", "\n")):
        block = block.strip()
        if not block:
            continue
        t = None
        payload_lines = []
        for line in block.splitlines():
            if t is None and "-->" in line:
                t = _cue_start_seconds(line)
                continue
            if "-->" in line:
                continue
            payload_lines.append(line)
        payload = " ".join(payload_lines)
        latlon = _latlon_from_payload(payload)
        if latlon is None:
            continue
        alt_m = _ALT_LABELLED.search(payload)
        fixes.append(GpsFix(lat=latlon[0], lon=latlon[1], timestamp_sec=t,
                            alt=float(alt_m.group(1)) if alt_m else None))
    return fixes
