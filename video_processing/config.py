"""
Shared configuration used by the DVPE preprocessing engine.

Only the thresholds DVPE relies on live here (validation, quality-metric,
duplicate, and scene defaults). DVPE-specific knobs — decode/quality-reject/
coverage/scoring/selection — live in ``video_processing/dvpe/config.py``.

Keeping these in one place means services stay parameter-free and testable.
"""
from dataclasses import dataclass, field
from pathlib import Path

# Repo root = the dvie_package/ directory (this file lives at video_processing/config.py).
# Keeping data under the package makes the pipeline runnable on any OS with no absolute
# paths baked in.
_PACKAGE_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class ValidationConfig:
    allowed_formats: tuple = (".mp4", ".avi", ".mov", ".mkv")
    allowed_resolutions: tuple = ((640, 360), (1280, 720), (1920, 1080), (2560, 1440), (3840, 2160))
    allowed_fps: tuple = (24, 25, 30, 50, 60)
    allowed_codecs: tuple = ("h264", "hevc", "h265")
    min_duration_sec: float = 1.0


@dataclass
class QualityConfig:
    blur_threshold: float = 60.0             # variance of Laplacian below this = too blurry
    min_brightness: float = 35.0
    max_brightness: float = 220.0
    min_contrast: float = 20.0
    max_noise_sigma: float = 25.0
    min_quality_score: float = 40.0          # frames scoring below this are dropped


@dataclass
class DuplicateConfig:
    phash_hamming_threshold: int = 5         # <= this distance => considered duplicate
    ssim_threshold: float = 0.94             # >= this similarity => considered duplicate
    orb_match_ratio_threshold: float = 0.85


@dataclass
class SceneConfig:
    hist_corr_threshold: float = 0.6         # below this correlation => new scene


@dataclass
class PipelineConfig:
    data_root: Path = field(default_factory=lambda: _PACKAGE_ROOT / "data")
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    quality: QualityConfig = field(default_factory=QualityConfig)
    duplicate: DuplicateConfig = field(default_factory=DuplicateConfig)
    scene: SceneConfig = field(default_factory=SceneConfig)


DEFAULT_CONFIG = PipelineConfig()
