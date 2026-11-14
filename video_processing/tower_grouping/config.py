"""
Configuration for the tower-grouping post-processing stage.

Every knob lives here (nothing hardcoded in the modules) and can be overridden from a
JSON file — same spirit as ``dvpe/config_io.py`` — so behaviour can be tuned per line /
per camera without editing code. Unknown keys are ignored with a warning, so a config
file cannot inject arbitrary attributes.

    {
      "switch_patience": 3,
      "max_assign_distance_m": 150.0,
      "interpolate_gps": true,
      "gps_max_time_gap_sec": 2.0,
      "move_frames": false,
      "folder_prefix": "Tower_",
      "folder_pad": 3
    }
"""
from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from video_processing.utils.io_helpers import get_logger

logger = get_logger("tower_grouping.config")


@dataclass
class TowerGroupingConfig:
    # --- assignment / switching -------------------------------------------------------
    switch_patience: int = 3
    """A newly-nearest tower must be nearest for this many CONSECUTIVE frames before the
    current-tower assignment switches to it. Prevents frames bouncing between towers on
    GPS noise (spec Step 5). 1 = switch immediately (no hysteresis)."""

    max_assign_distance_m: Optional[float] = None
    """If set, a frame whose nearest tower is farther than this (metres) is left
    UNASSIGNED (e.g. transit footage between towers). ``None`` = always assign the
    nearest tower, matching the base spec."""

    # --- GPS lookup -------------------------------------------------------------------
    interpolate_gps: bool = True
    """Resolve a frame's GPS by linearly interpolating between the two bracketing fixes
    (True) or by taking the single nearest-in-time fix (False). Exact frame-id matches in
    the GPS source always win over both."""

    gps_max_time_gap_sec: float = 2.0
    """If the nearest GPS fix (or bracketing fixes) is farther than this in time from a
    frame's timestamp, the frame is treated as having no reliable GPS and left
    unassigned. Guards against sparse/partial GPS tracks."""

    # --- output -----------------------------------------------------------------------
    move_frames: bool = False
    """Move frames into the tower folders (True) or copy them (False, default —
    non-destructive: the original ``best_frames/`` stays intact)."""

    folder_prefix: str = "Tower_"
    folder_pad: int = 3
    """Folder naming: ``<prefix><zero-padded number>`` e.g. ``Tower_001``. The number is
    parsed from the tower id (``T001`` -> ``001``); towers whose id has no trailing number
    fall back to order-of-appearance."""


DEFAULT_TOWER_GROUPING_CONFIG = TowerGroupingConfig()

_ALLOWED_KEYS = (
    "switch_patience", "max_assign_distance_m", "interpolate_gps",
    "gps_max_time_gap_sec", "move_frames", "folder_prefix", "folder_pad",
)


def apply_overrides(config: TowerGroupingConfig, data: dict) -> TowerGroupingConfig:
    """Apply a parsed config dict onto a TowerGroupingConfig in place (returns it)."""
    for key, value in data.items():
        if key.startswith("_"):          # allow "_comment" keys in the file
            continue
        if key in _ALLOWED_KEYS:
            setattr(config, key, value)
        else:
            logger.warning(f"Ignoring unknown config key: {key!r}")
    return config


def load_config(path: str, base: TowerGroupingConfig = None) -> TowerGroupingConfig:
    """Load a JSON config file and return a TowerGroupingConfig with overrides applied."""
    config = base or TowerGroupingConfig()
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Config file {path} must contain a JSON object at the top level.")
    logger.info(f"Loaded tower-grouping config overrides from {path}")
    return apply_overrides(config, data)
