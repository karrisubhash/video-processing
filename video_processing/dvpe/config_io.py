"""
Config file loading for DVPE (JSON, stdlib-only — no extra dependency).

Lets the tunable knobs (scoring weights, inspect meta-weights, adaptive-fps bands,
duplicate thresholds, coverage rule, selection mode, ...) live in a file so the system
can be tuned per camera / inspection condition without editing code. CLI flags applied
after this override the file.

Only known sections/keys are applied (unknown keys are ignored with a warning), so a
config file can't inject arbitrary attributes. Example:

    {
      "selection":      {"mode": "inspect"},
      "scoring":        {"weights": {"sharpness": 0.35, "contrast": 0.20,
                                     "brightness": 0.15, "exposure": 0.15, "noise": 0.15}},
      "inspect":        {"weights": {"quality": 0.4, "uniqueness": 0.25,
                                     "diversity": 0.2, "coverage": 0.15},
                         "hover_fps": 1, "move_fps": 4, "rotate_fps": 6,
                         "history_size": 8},
      "dedup_history_size": 8,
      "coverage":       {"max_gap_seconds": 1.0},
      "duplicate":      {"phash_hamming_threshold": 10, "ssim_threshold": 0.92}
    }
"""
from __future__ import annotations

import json
from pathlib import Path

from video_processing.dvpe.config import DVPEConfig, DEFAULT_DVPE_CONFIG
from video_processing.utils.io_helpers import get_logger

logger = get_logger("dvpe.config_io")

# section name -> (config attribute holding that section, allowed scalar keys)
_SECTIONS = {
    "selection": ("selection", ("mode", "keep_fps", "target_count", "target_fraction",
                                 "min_frame_gap", "min_output_frames", "max_output_frames")),
    "coverage": ("coverage", ("max_gap_seconds",)),
    "duplicate": ("duplicate", ("phash_hamming_threshold", "ssim_threshold",
                                "orb_match_ratio_threshold")),
    "decode": ("decode", ("max_decode_frames", "jpg_quality")),
    "quality_reject": ("quality_reject", ("extreme_blur_floor", "black_brightness",
                                          "white_brightness", "dark_fraction_max",
                                          "bright_fraction_max", "min_contrast", "noise_penalty_k")),
    "inspect": ("inspect", ("hover_fps", "move_fps", "rotate_fps", "hover_motion_max",
                            "move_motion_max", "history_size", "novelty_cap",
                            "diversity_motion_cap", "coverage_gap_cap", "reason_threshold")),
}
_TOP_LEVEL_KEYS = ("dedup_history_size", "keep_extracted_frames")


def apply_overrides(config: DVPEConfig, data: dict) -> DVPEConfig:
    """Apply a parsed config dict onto a DVPEConfig in place. Returns the same config."""
    for key in _TOP_LEVEL_KEYS:
        if key in data:
            setattr(config, key, data[key])

    for name, section in data.items():
        if name in _TOP_LEVEL_KEYS:
            continue
        # weights dicts (scoring + inspect) are merged, not replaced wholesale
        if name == "scoring" and isinstance(section, dict):
            _merge_weights(config.scoring.weights, section)
            continue
        if name == "inspect" and isinstance(section, dict) and "weights" in section:
            _merge_weights(config.inspect.weights, {"weights": section["weights"]})
        if name not in _SECTIONS:
            logger.warning(f"Ignoring unknown config section: {name!r}")
            continue
        attr, allowed = _SECTIONS[name]
        target = getattr(config, attr)
        for k, v in section.items():
            if k == "weights":
                continue  # handled above (inspect) / not applicable
            if k in allowed:
                setattr(target, k, v)
            else:
                logger.warning(f"Ignoring unknown key {name}.{k!r}")
    return config


def _merge_weights(target: dict, section: dict) -> None:
    weights = section.get("weights", section)  # accept {"weights": {...}} or flat {...}
    if isinstance(weights, dict):
        for k, v in weights.items():
            target[k] = float(v)


def load_config(path: str, base: DVPEConfig = None) -> DVPEConfig:
    """Load a JSON config file and return a DVPEConfig with those overrides applied
    (starting from a fresh DVPEConfig unless ``base`` is given)."""
    config = base or DVPEConfig()
    text = Path(path).read_text(encoding="utf-8")
    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError(f"Config file {path} must contain a JSON object at the top level.")
    logger.info(f"Loaded config overrides from {path}")
    return apply_overrides(config, data)
