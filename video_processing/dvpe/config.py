"""
Configuration for the DVPE preprocessing engine (maximal-retention model).

Philosophy: decode every frame, then remove ONLY frames we're confident add no value
(corrupted / completely black or white / extremely blurred / near-identical duplicates).
Everything else stays. A coverage safety rule guarantees temporal density, and no single
feature is trusted alone — blur + brightness + contrast gate quality; duplicate similarity
removes repeats; the time-interval rule guarantees coverage.

No enhancement, no detection.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from video_processing.config import _PACKAGE_ROOT, ValidationConfig, QualityConfig, DuplicateConfig


def _dvpe_validation_config() -> "ValidationConfig":
    """Lenient validation for a general preprocessing engine: accept ANY decodable video.
    We do NOT reject on frame rate, resolution, or codec — real drone clips have odd/variable
    frame rates (e.g. 18.6fps), portrait resolutions, and assorted codecs. Only genuinely
    unreadable/too-short files are rejected (the decode sanity check still runs)."""
    return ValidationConfig(
        allowed_formats=(".mp4", ".avi", ".mov", ".mkv", ".m4v", ".webm", ".mpg", ".mpeg", ".ts"),
        allowed_resolutions=((32, 32),),   # effectively accept any resolution
        allowed_fps=(),                    # skip the frame-rate gate
        allowed_codecs=(),                 # skip the codec gate (accept anything decodable)
        min_duration_sec=0.2,
    )


def _dvpe_duplicate_config() -> "DuplicateConfig":
    """Near-identical removal for DVPE: a slightly wider pHash prefilter (<=10) with SSIM
    (>=0.92) still confirming, so genuine near-identical frames are merged while distinct
    views are preserved (endorsed by the design review). Kept separate from DVIE's default."""
    return DuplicateConfig(phash_hamming_threshold=10, ssim_threshold=0.92)


@dataclass
class DecodeConfig:
    """Decode every frame. On extremely long/large videos, a stride keeps the candidate
    count (and disk) bounded (decode every ceil(total/max_decode_frames)-th frame)."""
    max_decode_frames: int = 3000
    jpg_quality: int = 95


@dataclass
class QualityRejectConfig:
    """Reject ONLY unusable frames — multi-signal, deliberately lenient. Everything that is
    not clearly worthless stays a candidate."""
    extreme_blur_floor: float = 12.0     # variance-of-Laplacian below this = extremely blurred
    black_brightness: float = 12.0       # mean brightness below this = ~completely black
    white_brightness: float = 243.0      # mean brightness above this = ~completely white
    dark_fraction_max: float = 0.985     # more than this fraction near-black = unusable
    bright_fraction_max: float = 0.985   # more than this fraction near-white = unusable
    min_contrast: float = 5.0            # std below this = no visible detail
    noise_penalty_k: float = 0.05        # noise-normalize sharpness so grain != "sharp"


@dataclass
class CoverageConfig:
    """Safety rule: never allow more than this many seconds without a kept frame, so
    temporal coverage is preserved even across long visually-static (deduped) stretches."""
    max_gap_seconds: float = 1.0


@dataclass
class DVPEScoringConfig:
    """Image-quality-only ranking (used to pick representatives when a gap must be filled
    or the soft output cap trims)."""
    weights: dict = field(default_factory=lambda: {
        "sharpness": 0.40, "contrast": 0.20, "brightness": 0.15, "noise": 0.15, "exposure": 0.10,
    })


@dataclass
class DVPESelectionConfig:
    """Density-capped by default: after bad/duplicate frames are removed, keep at most
    ``keep_fps`` frames per second — the highest-scoring representative in each short time
    window — so the output scales with duration instead of exploding to ~the whole video.
    The coverage safety rule still guarantees a lower bound (no gap > max_gap_seconds)."""
    mode: str = "auto"                   # 'auto' (density cap) | 'all' | 'count' | 'fraction' | 'inspect' (composite + adaptive)
    keep_fps: float = 2.0                # 'auto' mode: max kept frames per second of video
    target_count: Optional[int] = None
    target_fraction: float = 0.30
    min_frame_gap: int = 1
    min_output_frames: int = 1
    max_output_frames: int = 2000        # soft cap backstop for very long/dense videos


@dataclass
class InspectConfig:
    """Config for the inspection-aware 'inspect' selection mode (opt-in via --mode inspect).

    Redefines 'best frame' for inspection footage as
        High Quality + Unique Information + Good Coverage + Not Duplicate
    instead of highest image quality alone. A composite score combines four aspects
    (weights auto-normalised, so they need not sum to 1); the selection window is
    adaptive to camera motion so dynamic segments (drone moving/rotating) are sampled
    more densely than static ones (hovering); each kept frame records WHY it was kept.
    """
    # Meta-weights combining the four aspects of "best frame" (auto-normalised).
    weights: dict = field(default_factory=lambda: {
        "quality": 0.40,      # image quality (sharpness/contrast/brightness/noise/exposure)
        "uniqueness": 0.25,   # novelty vs the last `history_size` kept frames (pHash distance)
        "diversity": 0.20,    # camera movement since last kept frame (new viewpoint proxy)
        "coverage": 0.15,     # temporal gap since last kept frame (fill coverage)
    })
    # Adaptive selection window: camera-motion band -> target frames/sec.
    hover_fps: float = 1.0      # ~hovering: keep sparsely
    move_fps: float = 4.0       # moving: keep more
    rotate_fps: float = 6.0     # rotating/fast: keep the most
    hover_motion_max: float = 2.0   # mean-abs-diff (0-255) at/below this = ~hovering
    move_motion_max: float = 8.0    # at/below this = moving; above = rotating/fast
    # Uniqueness / diversity / coverage normalisation.
    history_size: int = 8              # compare against last N kept (also the dedup history depth)
    novelty_cap: float = 16.0          # pHash Hamming distance treated as "fully novel"
    diversity_motion_cap: float = 30.0 # cumulative motion since last kept treated as "fully new viewpoint"
    coverage_gap_cap: float = 1.0      # seconds since last kept treated as "fully coverage-critical"
    reason_threshold: float = 0.6      # a component (0-1) at/above this is listed as a keep reason


@dataclass
class DVPEConfig:
    data_root: Path = field(default_factory=lambda: _PACKAGE_ROOT / "data")
    keep_extracted_frames: bool = False
    dedup_history_size: int = 1                                                 # near-dup compare depth (1 = last kept only)
    validation: ValidationConfig = field(default_factory=_dvpe_validation_config)
    decode: DecodeConfig = field(default_factory=DecodeConfig)
    quality: QualityConfig = field(default_factory=QualityConfig)               # metric computation
    quality_reject: QualityRejectConfig = field(default_factory=QualityRejectConfig)
    duplicate: DuplicateConfig = field(default_factory=_dvpe_duplicate_config)
    coverage: CoverageConfig = field(default_factory=CoverageConfig)
    scoring: DVPEScoringConfig = field(default_factory=DVPEScoringConfig)
    selection: DVPESelectionConfig = field(default_factory=DVPESelectionConfig)
    inspect: InspectConfig = field(default_factory=InspectConfig)


DEFAULT_DVPE_CONFIG = DVPEConfig()
