"""
Tests for DVPE inspect mode (inspection-aware selection) and the JSON config loader.

- End-to-end on a synthetic clip with static + moving segments: asserts the composite
  score breakdown (quality/uniqueness/diversity/coverage), per-frame keep reasons, and a
  motion signal are all produced, and the flat package is written.
- Unit test for the motion -> target-fps mapping (adaptive window).
- Unit test for the JSON config loader (weights merge, section overrides, unknown-key safety).

    python -m pytest tests/test_dvpe_inspect.py -v
"""
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from video_processing.dvpe.config import DVPEConfig, InspectConfig
from video_processing.dvpe.config_io import apply_overrides, load_config
from video_processing.dvpe.pipeline import run_dvpe
from video_processing.dvpe.selection import _local_fps


@pytest.fixture
def motion_video(tmp_path):
    """8s clip: a bright rectangle that is static for a stretch, then sweeps across the
    frame (creating real frame-to-frame motion), over a textured background."""
    encoded = tmp_path / "motion.mp4"
    w, h, fps, seconds = 1280, 720, 30, 8
    writer = cv2.VideoWriter(str(encoded), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not writer.isOpened():
        pytest.skip("OpenCV could not open an mp4v VideoWriter in this environment.")
    rng = np.random.default_rng(11)
    bg = np.full((h, w, 3), 140, dtype=np.uint8)
    for k in range(0, h, 25):
        cv2.line(bg, (0, k), (w, k), (110, 110, 110), 1)
    n = fps * seconds
    for i in range(n):
        frame = bg.copy()
        # static for first third, then move the box across the frame
        t = max(0, i - n // 3)
        x = 60 + int((w - 300) * (t / (n - n // 3)))
        cv2.rectangle(frame, (x, 250), (x + 200, 500), (40, 40, 200), -1)
        frame = cv2.add(frame, rng.integers(0, 10, (h, w, 3), dtype=np.uint8))
        writer.write(frame)
    writer.release()
    return str(encoded)


def test_inspect_mode_end_to_end(motion_video, tmp_path):
    cfg = DVPEConfig()
    cfg.data_root = tmp_path / "data"
    cfg.validation.allowed_codecs = ("h264", "hevc", "h265", "mpeg4")
    cfg.selection.mode = "inspect"

    result = run_dvpe(motion_video, "TestInspect", cfg)

    assert result["accepted"] is True
    assert result["frames_kept"] > 0
    assert result["selection_mode"] == "inspect"
    assert result["largest_gap_seconds"] <= 1.5     # coverage safety rule still holds

    summ = json.loads(
        (tmp_path / "data" / "missions" / "TestInspect" / "processing_summary.json").read_text(encoding="utf-8"))
    entries = summ["best_frames"]
    assert len(entries) == result["frames_kept"]

    for e in entries:
        # every kept frame explains WHY it was kept, and carries a motion signal
        assert isinstance(e["reasons"], list) and len(e["reasons"]) >= 1
        assert e["motion"] is not None
        if e["coverage_forced"]:
            # force-kept by the safety rule (no composite score) — must say so
            assert "coverage" in e["reasons"]
        else:
            # composite breakdown present (not image-quality-only)
            for k in ("composite", "quality", "uniqueness", "diversity", "coverage"):
                assert k in e["score_breakdown"], f"missing {k} in breakdown"

    # at least one frame should be kept for a non-quality reason (the whole point of inspect)
    non_quality = {r for e in entries for r in e["reasons"]} - {"highest_quality"}
    assert non_quality, "expected some frames kept for uniqueness/coverage/movement"


def test_local_fps_bands():
    ci = InspectConfig(hover_fps=1.0, move_fps=4.0, rotate_fps=6.0,
                       hover_motion_max=2.0, move_motion_max=8.0)
    assert _local_fps(0.5, ci) == 1.0     # hovering
    assert _local_fps(2.0, ci) == 1.0     # boundary -> hover
    assert _local_fps(5.0, ci) == 4.0     # moving
    assert _local_fps(20.0, ci) == 6.0    # rotating/fast


def test_config_io_overrides(tmp_path):
    cfg = DVPEConfig()
    data = {
        "selection": {"mode": "inspect", "keep_fps": 3},
        "scoring": {"weights": {"sharpness": 0.35, "exposure": 0.15}},
        "inspect": {"weights": {"quality": 0.5, "uniqueness": 0.3},
                    "hover_fps": 2, "history_size": 10, "bogus": 1},
        "dedup_history_size": 8,
        "coverage": {"max_gap_seconds": 0.5},
        "duplicate": {"phash_hamming_threshold": 12},
        "totally_unknown_section": {"x": 1},
    }
    apply_overrides(cfg, data)

    assert cfg.selection.mode == "inspect"
    assert cfg.selection.keep_fps == 3
    assert cfg.scoring.weights["sharpness"] == 0.35
    assert cfg.scoring.weights["exposure"] == 0.15
    assert cfg.scoring.weights["contrast"] == 0.20     # untouched default preserved
    assert cfg.inspect.weights["quality"] == 0.5
    assert cfg.inspect.weights["uniqueness"] == 0.3
    assert cfg.inspect.hover_fps == 2
    assert cfg.inspect.history_size == 10
    assert not hasattr(cfg.inspect, "bogus")           # unknown key ignored, not injected
    assert cfg.dedup_history_size == 8
    assert cfg.coverage.max_gap_seconds == 0.5
    assert cfg.duplicate.phash_hamming_threshold == 12


def test_config_io_load_from_file(tmp_path):
    p = tmp_path / "weights.json"
    p.write_text(json.dumps({"selection": {"mode": "inspect"}}), encoding="utf-8")
    cfg = load_config(str(p))
    assert cfg.selection.mode == "inspect"
