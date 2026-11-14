"""
Frame -> tower assignment with GPS-noise hysteresis (spec Steps 3-6).

Per frame the nearest tower is the raw decision (``towers.nearest_tower``). Taken literally
that bounces between towers whenever GPS jitters near a boundary (spec Step 5). So instead of
switching on every raw flip, the assigner keeps a *current tower* and only switches once a
different tower has been nearest for ``switch_patience`` CONSECUTIVE frames — and when it does
switch, the frames of that confirming run are (retroactively) assigned to the new tower, while
isolated one-off flips stay with the current tower as noise.

Frames with no reliable GPS, or whose nearest tower is beyond ``max_assign_distance_m`` (e.g.
transit footage between towers), are left unassigned: they neither receive a tower nor disturb
the current/candidate state.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional

from video_processing.tower_grouping.config import TowerGroupingConfig
from video_processing.tower_grouping.towers import nearest_tower
from video_processing.utils.io_helpers import get_logger

logger = get_logger("tower_grouping.assign")


@dataclass
class FrameGeo:
    """One best frame with its resolved position (lat/lon None when GPS was unavailable)."""
    frame_id: str
    timestamp_sec: float
    file: str
    score: Optional[float] = None
    lat: Optional[float] = None
    lon: Optional[float] = None


@dataclass
class FrameAssignment:
    frame_id: str
    timestamp_sec: float
    file: str
    score: Optional[float]
    lat: Optional[float]
    lon: Optional[float]
    tower_id: Optional[str] = None          # final assignment after hysteresis
    distance_m: Optional[float] = None       # distance to the ASSIGNED tower
    nearest_tower_id: Optional[str] = None   # raw per-frame nearest (pre-hysteresis)
    nearest_distance_m: Optional[float] = None
    unassigned_reason: Optional[str] = None  # "no_gps" | "too_far" (when tower_id is None)


@dataclass
class AssignmentResult:
    assignments: list = field(default_factory=list)   # list[FrameAssignment] in timeline order
    switches: int = 0                                 # number of confirmed tower switches
    unassigned: int = 0


def assign_frames(frames: list, towers: list, config: TowerGroupingConfig) -> AssignmentResult:
    """Assign each :class:`FrameGeo` to a tower using the hysteresis switching rule.
    ``frames`` must be in timeline order."""
    n = len(frames)
    out = [FrameAssignment(f.frame_id, f.timestamp_sec, f.file, f.score, f.lat, f.lon)
           for f in frames]

    # Raw per-frame nearest tower (None when unusable: no GPS, or beyond the max distance).
    for i, f in enumerate(frames):
        if f.lat is None or f.lon is None:
            out[i].unassigned_reason = "no_gps"
            continue
        tower, dist = nearest_tower(f.lat, f.lon, towers)
        out[i].nearest_tower_id = tower.tower_id
        out[i].nearest_distance_m = round(dist, 3)
        if config.max_assign_distance_m is not None and dist > config.max_assign_distance_m:
            out[i].unassigned_reason = "too_far"   # instantaneous nearest still recorded above

    def instantaneous(i: int) -> Optional[str]:
        """The tower this frame counts as evidence for, or None if it must be ignored."""
        return None if out[i].unassigned_reason else out[i].nearest_tower_id

    current: Optional[str] = None
    candidate: Optional[str] = None
    run_indices: list = []
    switches = 0
    patience = max(1, int(config.switch_patience))

    for i in range(n):
        nb = instantaneous(i)
        if nb is None:
            continue                              # unknown/too-far: don't touch state
        if current is None:                       # first usable frame seeds the current tower
            current = nb
            out[i].tower_id = current
            continue
        if nb == current:
            out[i].tower_id = current
            candidate, run_indices = None, []     # positive evidence to stay resets any candidate
            continue

        # A different tower is nearest — accumulate a consecutive run before switching.
        if nb == candidate:
            run_indices.append(i)
        else:
            candidate, run_indices = nb, [i]
        out[i].tower_id = current                 # tentative: stay until the run confirms

        if len(run_indices) >= patience:          # confirmed: flip the whole run to candidate
            for j in run_indices:
                out[j].tower_id = candidate
            current = candidate
            switches += 1
            candidate, run_indices = None, []

    # Finalise assigned distances (distance to the tower each frame actually landed in).
    tower_by_id = {t.tower_id: t for t in towers}
    unassigned = 0
    for a in out:
        if a.tower_id is None:
            unassigned += 1
            a.unassigned_reason = a.unassigned_reason or "no_gps"
            continue
        a.unassigned_reason = None
        if a.tower_id == a.nearest_tower_id:
            a.distance_m = a.nearest_distance_m
        elif a.lat is not None and a.lon is not None and a.tower_id in tower_by_id:
            from video_processing.tower_grouping.towers import haversine_m
            t = tower_by_id[a.tower_id]
            a.distance_m = round(haversine_m(a.lat, a.lon, t.lat, t.lon), 3)

    logger.info(f"Assigned {n - unassigned}/{n} frame(s) across towers with {switches} switch(es); "
                f"{unassigned} unassigned")
    return AssignmentResult(assignments=out, switches=switches, unassigned=unassigned)
