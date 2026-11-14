"""
APTRANSCO tower database + geodesy (spec Steps 1 and 3).

The tower database is a fixed JSON list of transmission-tower coordinates that changes only
when towers are added:

    [ {"tower_id": "T001", "lat": 17.734820, "lon": 83.281210}, ... ]

``nearest_tower`` compares a frame's GPS against every tower with the Haversine great-circle
distance and returns the closest one — the raw per-frame decision the switching rule in
``assign.py`` then smooths.
"""
from __future__ import annotations
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from video_processing.tower_grouping.gps import _get, _to_float, _LAT_KEYS, _LON_KEYS
from video_processing.utils.io_helpers import get_logger

logger = get_logger("tower_grouping.towers")

_EARTH_RADIUS_M = 6_371_000.0
_ID_KEYS = ("tower_id", "id", "name", "tower")


@dataclass
class Tower:
    tower_id: str
    lat: float
    lon: float


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance between two lat/lon points, in metres (spec Step 3)."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2.0) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2.0) ** 2
    return 2.0 * _EARTH_RADIUS_M * math.asin(min(1.0, math.sqrt(a)))


def load_tower_db(path: str) -> list:
    """Load the tower database (JSON list). Accepts lat/latitude and lon/longitude keys and
    a ``{"towers": [...]}`` wrapper. Raises on an empty/invalid database."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Tower database not found: {path}")
    data = json.loads(p.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        for v in data.values():
            if isinstance(v, list):
                data = v
                break
    if not isinstance(data, list):
        raise ValueError("Tower database must be a JSON list of towers (or an object containing one).")

    towers, seen = [], set()
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            continue
        lat, lon = _to_float(_get(item, _LAT_KEYS)), _to_float(_get(item, _LON_KEYS))
        if lat is None or lon is None:
            logger.warning(f"Skipping tower entry #{i} with no usable lat/lon: {item}")
            continue
        tid = _get(item, _ID_KEYS)
        tid = str(tid) if tid is not None else f"T{i + 1:03d}"
        if tid in seen:
            logger.warning(f"Duplicate tower_id {tid!r}; keeping the first occurrence")
            continue
        seen.add(tid)
        towers.append(Tower(tower_id=tid, lat=lat, lon=lon))

    if not towers:
        raise ValueError(f"Tower database {path} contained no usable towers.")
    logger.info(f"Loaded {len(towers)} tower(s) from {p.name}")
    return towers


def nearest_tower(lat: float, lon: float, towers: list) -> tuple:
    """Return ``(Tower, distance_m)`` for the closest tower to the given position."""
    best, best_d = None, math.inf
    for tower in towers:
        d = haversine_m(lat, lon, tower.lat, tower.lon)
        if d < best_d:
            best, best_d = tower, d
    return best, best_d
