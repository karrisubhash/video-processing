"""
Tests for the DVPE preprocessing engine (density-capped, maximal-usable model).

- End-to-end smoke test on a synthetic clip: decode-all -> lenient quality -> near-dup
  removal -> density-capped selection -> coverage fill. Asserts a flat package with NO
  enhancement (native resolution), NO detection artifacts, coverage metrics present, and a
  duration-proportionate frame count (not the whole video).
- Unit test for the coverage safety rule (no gap larger than the limit).
- Unit test for 'all' mode keeping every usable, non-duplicate frame.

    python -m pytest tests/test_dvpe_smoke.py -v
"""
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from video_processing.dvpe.config import DVPEConfig, DVPESelectionConfig, CoverageConfig
from video_processing.dvpe.pipeline import run_dvpe
from video_processing.dvpe.selection import select_best_frames
from video_processing.dvpe.coverage import fill_gaps, largest_gap_frames
from video_processing.models.schema import FrameRecord


@pytest.fixture
def synthetic_video(tmp_path):
    """A new random scene every 5 frames with near-identical noisy frames in between."""
    encoded = tmp_path / "synthetic.mp4"
    w, h, fps, seconds = 1280, 720, 30, 8
    writer = cv2.VideoWriter(str(encoded), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not writer.isOpened():
        pytest.skip("OpenCV could not open an mp4v VideoWriter in this environment.")
    rng = np.random.default_rng(7)
    n_frames = fps * seconds
    base = None
    for i in range(n_frames):
        if i % 5 == 0:
            base = np.full((h, w, 3), int(rng.integers(120, 190)), dtype=np.uint8)
            for _ in range(4):
                x1, y1 = int(rng.integers(0, w - 260)), int(rng.integers(0, h - 260))
                x2, y2 = x1 + int(rng.integers(90, 250)), y1 + int(rng.integers(90, 250))
                color = tuple(int(v) for v in rng.integers(30, 100, 3))
                cv2.rectangle(base, (x1, y1), (x2, y2), color, -1)
            for k in range(0, h, 40):
                cv2.line(base, (0, k), (w, k), (150, 150, 150), 1)
        frame = cv2.add(base, rng.integers(0, 12, (h, w, 3), dtype=np.uint8))
        writer.write(frame)
    writer.release()
    return str(encoded)


def test_dvpe_end_to_end(synthetic_video, tmp_path):
    cfg = DVPEConfig()
    cfg.data_root = tmp_path / "data"
    cfg.validation.allowed_codecs = ("h264", "hevc", "h265", "mpeg4")

    result = run_dvpe(synthetic_video, "TestDVPE", cfg)

    assert result["accepted"] is True
    # Duration-proportionate: an 8s clip at the default ~2 fps keeps roughly 2*8=16, not ~all.
    assert 6 <= result["frames_kept"] <= 40
    assert result["frames_decoded"] > result["frames_kept"]
    for key in ("original_frame_count", "frames_rejected_by_quality",
                "frames_removed_as_duplicates", "largest_gap_frames", "largest_gap_seconds"):
        assert key in result
    assert result["largest_gap_seconds"] <= 1.5     # coverage safety rule

    mission = tmp_path / "data" / "missions" / "TestDVPE"
    assert (mission / "metadata.json").exists()
    assert (mission / "processing_summary.json").exists()

    best_files = sorted((mission / "best_frames").glob("*.jpg"))
    assert len(best_files) == result["frames_kept"]
    assert cv2.imread(str(best_files[0])).shape[:2] == (720, 1280)   # native res, no enhancement

    summ = json.loads((mission / "processing_summary.json").read_text(encoding="utf-8"))
    assert "tower_tracks" not in summ and "detections" not in summ
    assert len(summ["best_frames"]) == result["frames_kept"]
    for e in summ["best_frames"]:
        assert e["written"] is True
        assert "coverage_forced" in e


def test_coverage_safety_rule():
    """No gap larger than max_gap even when selection left two far-apart frames."""
    records = [FrameRecord(frame_id=f"f{i:03d}", frame_index=i, path="",
                           passed_quality=True) for i in range(100)]
    for r in records:
        r.sharpness = float(r.frame_index)
    selected = [records[0], records[90]]                     # 90 apart

    selected, forced = fill_gaps(selected, records, fps=30.0,
                                 config=CoverageConfig(max_gap_seconds=1.0))   # gap <= 30
    assert forced >= 2
    assert largest_gap_frames(selected) <= 30


def test_all_mode_keeps_every_usable():
    records = []
    for i in range(20):
        r = FrameRecord(frame_id=f"f{i:02d}", frame_index=i * 4, path="", passed_quality=True)
        r.is_duplicate = i in (5, 9)
        r.final_score = 50 + i
        records.append(r)

    cfg = DVPESelectionConfig(mode="all", min_output_frames=1, max_output_frames=2000)
    selected = select_best_frames(records, cfg, fps=30.0)
    assert len(selected) == 18                                # 20 - 2 duplicates
    assert all(not r.is_duplicate for r in selected)
