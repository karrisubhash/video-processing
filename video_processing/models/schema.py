"""
Shared data structures passed between pipeline layers.
Keeping these in one place means every service imports the same shapes —
no layer invents its own dict schema.
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class BBox:
    x1: float
    y1: float
    x2: float
    y2: float

    def area(self) -> float:
        return max(0.0, self.x2 - self.x1) * max(0.0, self.y2 - self.y1)

    def iou(self, other: "BBox") -> float:
        xa, ya = max(self.x1, other.x1), max(self.y1, other.y1)
        xb, yb = min(self.x2, other.x2), min(self.y2, other.y2)
        inter = max(0.0, xb - xa) * max(0.0, yb - ya)
        union = self.area() + other.area() - inter
        return inter / union if union > 0 else 0.0

    def to_dict(self):
        return asdict(self)


@dataclass
class Detection:
    label: str
    bbox: BBox
    confidence: float
    track_id: Optional[int] = None

    def to_dict(self):
        d = asdict(self)
        return d


@dataclass
class FrameRecord:
    """One extracted frame and everything computed about it as it moves through the pipeline."""
    frame_id: str
    frame_index: int              # index in original video
    path: str = ""                # populated once written to disk
    timestamp_sec: float = 0.0

    # Layer 4 — quality
    quality_score: Optional[float] = None
    sharpness: Optional[float] = None
    brightness: Optional[float] = None
    contrast: Optional[float] = None
    noise_sigma: Optional[float] = None
    overexposed: bool = False
    underexposed: bool = False
    dark_fraction: Optional[float] = None    # fraction of near-black pixels (exposure detail)
    bright_fraction: Optional[float] = None  # fraction of near-white pixels
    passed_quality: bool = False

    # Layer 5 — duplicate removal
    phash: Optional[str] = None
    is_duplicate: bool = False
    duplicate_of: Optional[str] = None

    # Layer 6 — scene
    scene_id: Optional[str] = None
    is_scene_change: bool = False   # first frame of a new scene (DVPE always keeps these)
    keep_reasons: list = field(default_factory=list)  # DVPE keep-gate: which criteria fired

    # DVPE inspect mode — camera dynamics (mean-abs-diff to previous decoded frame)
    motion: Optional[float] = None

    # Layer 7/8 — detection & tracking
    detections: list = field(default_factory=list)   # list[Detection]

    # Layer 9 — scoring
    final_score: Optional[float] = None
    score_breakdown: dict = field(default_factory=dict)

    # Layer 10 — best view
    assigned_view: Optional[str] = None
    selected_as_best: bool = False

    def to_dict(self):
        d = asdict(self)
        d["detections"] = [det.to_dict() if isinstance(det, Detection) else det for det in self.detections]
        return d


@dataclass
class MissionMetadata:
    """Layer 2 output. Whatever isn't available from the file/EXIF/flight-log stays None."""
    mission_id: str
    video_name: str
    drone_name: Optional[str] = None
    camera_model: Optional[str] = None
    resolution: Optional[tuple] = None
    fps: Optional[float] = None
    duration_sec: Optional[float] = None
    codec: Optional[str] = None
    frame_count: Optional[int] = None
    gps: Optional[dict] = None
    altitude: Optional[float] = None
    heading: Optional[float] = None
    yaw: Optional[float] = None
    pitch: Optional[float] = None
    roll: Optional[float] = None
    capture_time: Optional[str] = None
    flight_speed: Optional[float] = None
    camera_angle: Optional[float] = None

    def to_dict(self):
        return asdict(self)
