"""
Layer 1 — Video Validation.

Verifies an uploaded drone video is decodable and meets format/resolution/fps/codec
requirements before anything downstream touches it. Uses the shared probe (ffprobe
when installed, OpenCV otherwise) for container/codec info, plus an OpenCV decode
sanity check.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2

from video_processing.config import ValidationConfig, DEFAULT_CONFIG
from video_processing.utils.io_helpers import get_logger
from video_processing.utils.probe import probe_video

logger = get_logger("layer1.validator")


@dataclass
class ValidationResult:
    accepted: bool
    reasons: list = field(default_factory=list)
    probe: Optional[dict] = None

    def to_dict(self):
        return {"accepted": self.accepted, "reasons": self.reasons, "probe": self.probe}


def _can_decode_first_frame(path: Path) -> bool:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        cap.release()
        return False
    ok, _ = cap.read()
    cap.release()
    return ok


def validate_video(video_path: str, config: ValidationConfig = None) -> ValidationResult:
    """Run Layer 1 checks. Returns a ValidationResult; does not raise on rejection."""
    config = config or DEFAULT_CONFIG.validation
    path = Path(video_path)
    reasons = []

    if not path.exists():
        return ValidationResult(accepted=False, reasons=[f"File not found: {video_path}"])

    if path.suffix.lower() not in config.allowed_formats:
        reasons.append(f"Unsupported format '{path.suffix}'. Allowed: {config.allowed_formats}")

    probe = probe_video(str(path))
    if probe is None:
        reasons.append("Could not read video stream metadata (corrupt or unsupported container).")
    else:
        codec = (probe.get("codec_name") or "").lower()
        if config.allowed_codecs and codec and codec not in config.allowed_codecs:
            reasons.append(f"Unsupported codec '{codec}'. Allowed: {config.allowed_codecs}")

        width, height = probe.get("width"), probe.get("height")
        if width and height:
            # Compare orientation-independently: a 360x640 portrait clip is the same capture
            # quality as an allowed 640x360 landscape one, so gate on short/long side rather
            # than width>=min_width AND height>=min_height (which wrongly rejects portrait).
            allowed_short = min(min(w, h) for w, h in config.allowed_resolutions)
            allowed_long = min(max(w, h) for w, h in config.allowed_resolutions)
            short_side, long_side = min(width, height), max(width, height)
            if short_side < allowed_short or long_side < allowed_long:
                reasons.append(
                    f"Resolution {width}x{height} below minimum "
                    f"{allowed_long}x{allowed_short} (orientation-independent)."
                )

        fps = probe.get("fps")
        if fps and config.allowed_fps:
            if not any(abs(fps - a) <= 1.0 for a in config.allowed_fps):
                reasons.append(f"FPS {fps} not in allowed set {config.allowed_fps}.")

        duration = probe.get("duration")
        try:
            if duration is not None and float(duration) < config.min_duration_sec:
                reasons.append(f"Duration {duration}s below minimum {config.min_duration_sec}s.")
        except (ValueError, TypeError):
            pass

    if not _can_decode_first_frame(path):
        reasons.append("Video failed OpenCV decode sanity check (possible corruption).")

    accepted = len(reasons) == 0
    logger.info(f"{'ACCEPTED' if accepted else 'REJECTED'}: {path.name} ({len(reasons)} issue(s))")
    return ValidationResult(accepted=accepted, reasons=reasons, probe=probe)
