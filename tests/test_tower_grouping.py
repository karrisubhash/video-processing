"""
Tests for the tower-grouping post-processing stage.

Covers the geodesy (Haversine + nearest), the GPS-noise hysteresis switching rule, the
three GPS source parsers (SRT / JSON / CSV) with timestamp interpolation, and an
end-to-end grouping run on a temp mission fixture.

    python -m pytest tests/test_tower_grouping.py -v
"""
import json
from pathlib import Path

import pytest

from video_processing.tower_grouping.assign import FrameGeo, assign_frames
from video_processing.tower_grouping.config import TowerGroupingConfig, load_config
from video_processing.tower_grouping.gps import load_gps_track
from video_processing.tower_grouping.grouping import run_tower_grouping
from video_processing.tower_grouping.towers import Tower, haversine_m, load_tower_db, nearest_tower


# --------------------------------------------------------------------------------------
# Geodesy
# --------------------------------------------------------------------------------------
def test_haversine_known_distances():
    assert haversine_m(0, 0, 0, 0) == 0.0
    # One degree of latitude is ~111.19 km anywhere on the globe.
    assert haversine_m(0, 0, 1, 0) == pytest.approx(111_195, rel=0.01)
    # The spec's ~8 m example: drone (17.73492, 83.28131) vs tower (17.73482, 83.28121).
    d = haversine_m(17.73492, 83.28131, 17.73482, 83.28121)
    assert 5 <= d <= 20


def test_nearest_tower_picks_closest():
    towers = [Tower("T001", 0.0, 0.0), Tower("T002", 0.0, 1.0), Tower("T003", 0.0, 2.0)]
    t, d = nearest_tower(0.0, 0.9, towers)
    assert t.tower_id == "T002"
    assert d == pytest.approx(haversine_m(0.0, 0.9, 0.0, 1.0))


# --------------------------------------------------------------------------------------
# Hysteresis switching (spec Step 5)
# --------------------------------------------------------------------------------------
_TOWERS = [Tower("T001", 0.0, 0.0), Tower("T002", 0.0, 1.0)]


def _frames(codes):
    """Build FrameGeo list; 'A' snaps to T001 (0,0), 'B' to T002 (0,1), '.' has no GPS."""
    coords = {"A": (0.0, 0.0), "B": (0.0, 1.0)}
    out = []
    for i, c in enumerate(codes):
        lat, lon = coords.get(c, (None, None))
        out.append(FrameGeo(frame_id=f"frame{i:05d}", timestamp_sec=float(i),
                            file=f"best_frames/frame{i:05d}.jpg", score=float(i), lat=lat, lon=lon))
    return out


def test_isolated_flips_are_noise():
    """A single opposite-tower flip must NOT switch (spec Step 5)."""
    res = assign_frames(_frames("AABABAA"), _TOWERS, TowerGroupingConfig(switch_patience=3))
    assert [a.tower_id for a in res.assignments] == ["T001"] * 7
    assert res.switches == 0


def test_sustained_run_switches_and_reassigns_run():
    """Once a new tower is nearest for `switch_patience` frames, switch AND assign the
    confirming run to the new tower."""
    res = assign_frames(_frames("AABBB"), _TOWERS, TowerGroupingConfig(switch_patience=3))
    assert [a.tower_id for a in res.assignments] == ["T001", "T001", "T002", "T002", "T002"]
    assert res.switches == 1


def test_patience_one_switches_immediately():
    res = assign_frames(_frames("AAB"), _TOWERS, TowerGroupingConfig(switch_patience=1))
    assert [a.tower_id for a in res.assignments] == ["T001", "T001", "T002"]


def test_no_gps_and_too_far_are_unassigned():
    # No-GPS frame in the middle does not disturb the current tower.
    res = assign_frames(_frames("AA.AA"), _TOWERS, TowerGroupingConfig(switch_patience=3))
    towers = [a.tower_id for a in res.assignments]
    assert towers == ["T001", "T001", None, "T001", "T001"]
    assert res.unassigned == 1
    assert res.assignments[2].unassigned_reason == "no_gps"

    # A frame far from every tower with a max-distance cap is left unassigned.
    far = [FrameGeo("frame00000", 0.0, "best_frames/frame00000.jpg", 1.0, lat=45.0, lon=45.0)]
    res2 = assign_frames(far, _TOWERS, TowerGroupingConfig(max_assign_distance_m=100.0))
    assert res2.assignments[0].tower_id is None
    assert res2.assignments[0].unassigned_reason == "too_far"
    assert res2.assignments[0].nearest_tower_id is not None   # raw nearest still recorded


# --------------------------------------------------------------------------------------
# GPS source parsers
# --------------------------------------------------------------------------------------
def test_gps_json_spec_format(tmp_path):
    """The exact shape from spec Step 2."""
    data = [
        {"frame": "frame00120.jpg", "timestamp": 5.00, "latitude": 17.73492, "longitude": 83.28131},
        {"frame": "frame00121.jpg", "timestamp": 5.10, "latitude": 17.73495, "longitude": 83.28135},
    ]
    p = tmp_path / "track.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    track = load_gps_track(str(p))
    assert len(track) == 2
    # Exact frame-id match wins.
    s = track.lookup(0.0, frame_id="frame00120")
    assert s.source == "frame_id" and s.lat == pytest.approx(17.73492)


def test_gps_json_interpolation(tmp_path):
    data = [{"timestamp": 0.0, "lat": 0.0, "lon": 0.0},
            {"timestamp": 10.0, "lat": 10.0, "lon": 20.0}]
    p = tmp_path / "t.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    track = load_gps_track(str(p))
    s = track.lookup(5.0, interpolate=True)
    assert s.source == "interpolated"
    assert s.lat == pytest.approx(5.0) and s.lon == pytest.approx(10.0)
    # Nearest-in-time mode.
    s2 = track.lookup(9.0, interpolate=False, max_time_gap_sec=2.0)
    assert s2.source == "nearest" and s2.lat == pytest.approx(10.0)
    # Beyond the allowed time gap -> no fix.
    assert track.lookup(100.0, interpolate=True, max_time_gap_sec=2.0) is None


def test_gps_csv(tmp_path):
    p = tmp_path / "t.csv"
    p.write_text("timestamp,latitude,longitude\n0.0,17.7,83.2\n1.0,17.8,83.3\n", encoding="utf-8")
    track = load_gps_track(str(p))
    assert len(track) == 2
    s = track.lookup(0.5, interpolate=True)
    assert s.lat == pytest.approx(17.75)


def test_gps_srt_labelled(tmp_path):
    srt = (
        "1\n00:00:00,000 --> 00:00:01,000\n"
        "<font size=\"28\">[latitude: 17.734820] [longitude: 83.281210] [rel_alt: 50.0]</font>\n\n"
        "2\n00:00:01,000 --> 00:00:02,000\n"
        "[latitude: 17.735450] [longitude: 83.282890] [rel_alt: 51.0]\n"
    )
    p = tmp_path / "dji.srt"
    p.write_text(srt, encoding="utf-8")
    track = load_gps_track(str(p))
    assert len(track) == 2
    s = track.lookup(0.0, interpolate=True)
    assert s.lat == pytest.approx(17.734820) and s.lon == pytest.approx(83.281210)


def test_gps_srt_gps_tuple_order_corrected(tmp_path):
    """DJI GPS(lon,lat,alt); a magnitude check must keep latitude in [-90,90]."""
    srt = ("1\n00:00:00,000 --> 00:00:01,000\nGPS(83.281210,17.734820,50.0)\n")
    p = tmp_path / "dji2.srt"
    p.write_text(srt, encoding="utf-8")
    track = load_gps_track(str(p))
    s = track.lookup(0.0, interpolate=True)
    assert -90 <= s.lat <= 90
    assert s.lat == pytest.approx(17.734820) and s.lon == pytest.approx(83.281210)


def test_tower_db_loads_and_aliases(tmp_path):
    p = tmp_path / "towers.json"
    p.write_text(json.dumps([
        {"tower_id": "T001", "lat": 17.734820, "lon": 83.281210},
        {"tower_id": "T002", "latitude": 17.735450, "longitude": 83.282890},
    ]), encoding="utf-8")
    towers = load_tower_db(str(p))
    assert [t.tower_id for t in towers] == ["T001", "T002"]
    assert towers[1].lat == pytest.approx(17.735450)


def test_config_json_overrides(tmp_path):
    p = tmp_path / "cfg.json"
    p.write_text(json.dumps({"switch_patience": 5, "move_frames": True, "bogus": 1}), encoding="utf-8")
    cfg = load_config(str(p))
    assert cfg.switch_patience == 5 and cfg.move_frames is True


# --------------------------------------------------------------------------------------
# End-to-end grouping on a temp mission
# --------------------------------------------------------------------------------------
def _make_mission(tmp_path):
    mdir = tmp_path / "Mission_TEST"
    (mdir / "best_frames").mkdir(parents=True)
    entries, gps = [], []
    # Frames 0-2 near T001 (0,0); frames 3-5 near T002 (0,1). Scores peak at frame 2 and 5.
    coords = [(0.0, 0.0)] * 3 + [(0.0, 1.0)] * 3
    scores = [10, 20, 99, 30, 40, 98]
    for i, ((lat, lon), score) in enumerate(zip(coords, scores)):
        fid = f"frame{i:05d}"
        (mdir / "best_frames" / f"{fid}.jpg").write_bytes(b"\xff\xd8\xff\xd9")
        entries.append({"frame_id": fid, "frame_index": i, "timestamp_sec": float(i),
                        "file": f"best_frames/{fid}.jpg", "score": float(score)})
        gps.append({"frame": f"{fid}.jpg", "timestamp": float(i), "latitude": lat, "longitude": lon})
    (mdir / "processing_summary.json").write_text(
        json.dumps({"mission_id": "Mission_TEST", "best_frames": entries}), encoding="utf-8")

    towers_p = tmp_path / "towers.json"
    towers_p.write_text(json.dumps([
        {"tower_id": "T001", "lat": 0.0, "lon": 0.0},
        {"tower_id": "T002", "lat": 0.0, "lon": 1.0},
    ]), encoding="utf-8")
    gps_p = tmp_path / "gps.json"
    gps_p.write_text(json.dumps(gps), encoding="utf-8")
    return mdir, towers_p, gps_p


def test_end_to_end_grouping(tmp_path):
    mdir, towers_p, gps_p = _make_mission(tmp_path)
    summary = run_tower_grouping(str(towers_p), str(gps_p), mission_dir=str(mdir),
                                 config=TowerGroupingConfig(switch_patience=3))

    assert summary["frames_total"] == 6
    assert summary["frames_assigned"] == 6
    assert summary["towers_populated"] == 2
    assert summary["tower_switches"] == 1

    t1 = mdir / "Tower_001"
    t2 = mdir / "Tower_002"
    assert t1.is_dir() and t2.is_dir()
    assert len(list(t1.glob("*.jpg"))) == 3
    assert len(list(t2.glob("*.jpg"))) == 3

    m1 = json.loads((t1 / "metadata.json").read_text(encoding="utf-8"))
    assert m1["tower_id"] == "T001"
    assert m1["frames"] == 3
    assert m1["best_frame"] == "frame00002.jpg"      # highest score in the group
    assert m1["start_time"] == 0.0 and m1["end_time"] == 2.0
    m2 = json.loads((t2 / "metadata.json").read_text(encoding="utf-8"))
    assert m2["best_frame"] == "frame00005.jpg"

    # Copy mode is non-destructive: best_frames/ stays intact.
    assert len(list((mdir / "best_frames").glob("*.jpg"))) == 6
    assert (mdir / "tower_grouping_summary.json").exists()


def test_end_to_end_move_mode(tmp_path):
    mdir, towers_p, gps_p = _make_mission(tmp_path)
    run_tower_grouping(str(towers_p), str(gps_p), mission_dir=str(mdir),
                       config=TowerGroupingConfig(switch_patience=3, move_frames=True))
    # Move mode empties best_frames/ (all six frames relocated into tower folders).
    assert len(list((mdir / "best_frames").glob("*.jpg"))) == 0
    assert len(list((mdir / "Tower_001").glob("*.jpg"))) == 3
    assert len(list((mdir / "Tower_002").glob("*.jpg"))) == 3
