# APTRANSCO Drone Video Preprocessing Engine (DVPE)

**Version:** 2.0
**Updated:** 2026‑07‑20
**Status:** ✅ *Working end‑to‑end, Windows‑verified on real footage.*

DVPE is a **preprocessing‑only** engine: raw drone video → a small set of the best,
non‑duplicate, quality‑checked frames + metadata. It contains **no
detection/tracking/YOLO/SAM/enhancement** — that is left to whatever inspection
system consumes the `best_frames/` output.

> **Note:** this repository previously also shipped a full detection engine (DVIE — 12‑layer
> YOLO/tracking/enhancement pipeline). DVIE has been removed; this repo is now DVPE‑only. See §9.

> This document is the single source of truth for the project: how the engine works, how to
> run it, how it's configured, what's verified, and how to extend it.

---

## 1. The problem this solves

The naive workflow — *drone video → extract random frames → run AI → report* — produces
thousands of duplicate frames, motion blur, poor lighting, inconsistent angles, huge compute
cost, and a low‑quality dataset that makes models slow and inaccurate.

DVPE inserts an intelligence layer between the raw video and the AI:

```
Drone Video → [ DVPE preprocessing ] → Clean best frames → YOLO → SAM → Analytics
```

DVPE answers *"give me the best usable frames from any video, and nothing junk"* —
general‑purpose, model‑free, works on any clip.

---

## 2. The engine at a glance

| | **DVPE** (preprocessing) |
|---|---|
| Goal | Best usable frames from any video |
| Detection / tracking | ❌ none (by design) |
| Enhancement | ❌ frames copied unmodified |
| Frame strategy | **Decode every frame**, reject only unusable, remove only near‑identical, density‑cap the rest |
| Output | Flat `best_frames/` + `metadata.json` + `processing_summary.json` |
| Validation | Lenient — accepts any decodable video |
| Entry point | `python -m video_processing.dvpe.pipeline` |
| Status | ✅ Working, verified on real footage |

**Design principle:** no single feature is ever trusted alone — blur **+** brightness **+**
contrast gate quality; perceptual‑hash **+** SSIM confirm duplicates; a time‑interval coverage
rule guarantees temporal density.

### Per‑stage status

| Stage | Status | Notes |
|-------|--------|-------|
| Validation (lenient) | ✅ | Accepts any decodable video — no fps/resolution/codec gate; only unreadable/too‑short rejected. |
| Metadata | ✅ | name/duration/fps/resolution/frame_count/codec/file_size/capture_time. |
| Decode‑every‑frame | ✅ | Metrics computed **inline** (no disk re‑read). Stride cap on very long videos (`max_decode_frames=3000`). |
| Quality analysis (reject only unusable) | ✅ | Drops only corrupted / all‑black / all‑white / no‑contrast / extremely‑blurred (noise‑normalized). |
| Near‑duplicate removal | ✅ | In‑memory pHash(≤10)+SSIM(≥0.92); keeps the sharpest of a cluster. |
| Best‑frame selection | ✅ | Density cap: keep ≤ `keep_fps` (default 2) frames/sec — highest‑scoring per window. Modes: auto/all/count/fraction. |
| Coverage safety rule | ✅ | Force‑keeps frames so no gap between kept frames exceeds `max_gap_seconds` (default 1.0s). |
| Output | ✅ | Copies selected frames unmodified; writes flat package; deletes frame cache unless `--keep-frames`. |

---

## 3. Architecture

```
Upload/session → Validation (lenient) → Metadata → Decode EVERY frame →
Quality Analysis (reject ONLY unusable) → Near‑Duplicate Removal (only near‑identical) →
Frame Scoring (image quality only) → Best‑Frame Selection (density cap) →
Coverage Check (no gap > max_gap_seconds) → Output (flat package)
```

**Philosophy:** decode every frame, then remove *only* what's clearly worthless (corrupted /
all‑black / all‑white / no‑contrast / extremely blurred) or redundant (near‑identical), then cap
density so the output scales with the video's real content instead of exploding to ~the whole
clip. A coverage safety rule guarantees a lower bound on temporal density.

**Data‑flow contract:** every stage reads/writes a shared `FrameRecord` object (see §5) — no
stage invents its own dict schema. Frames flow as a list; stages mutate records in place and the
orchestrator filters between stages.

---

## 4. Modules · `video_processing/dvpe/`

- **`pipeline.py`** — orchestrator + CLI (`run_dvpe`). Times every stage into `timings_sec`.
- **`config.py`** — `DVPEConfig` with lenient validation, `DecodeConfig`, `QualityRejectConfig`, `DuplicateConfig` (pHash≤10/SSIM≥0.92), `CoverageConfig`, `DVPEScoringConfig`, `DVPESelectionConfig`.
- **`metadata.py`** — video name/duration/fps/resolution/frame_count/codec/file_size + best‑effort container `capture_time` (no mtime fallback).
- **`extraction.py`** — `decode_all_frames`: writes every frame (stride only if `total > max_decode_frames`) and computes **all quality metrics + pHash + a 128‑px grayscale thumbnail inline**, so no later stage re‑reads from disk.
- **`quality_gate.py`** — `apply_quality`: rejects only unusable frames; sharpness is **noise‑normalized** so high‑ISO grain isn't mistaken for detail. Records a per‑frame `rejected:<reason>` in `keep_reasons`.
- **`dedup.py`** — `remove_near_duplicates`: in‑memory pHash prefilter + SSIM confirm against the last kept frame; keeps the **sharpest** of a near‑identical cluster. Uses the stashed thumbnail → one of the fastest stages instead of the slowest.
- **`scoring.py`** — `score_frames`: image‑quality‑only 0–100 (`sharpness .40, contrast .20, brightness .15, noise .15, exposure .10`). No geometry/detection signals.
- **`selection.py`** — `select_best_frames`: density cap (best‑scoring frame per `1/keep_fps`‑second window) in `auto`; also `all` / `count` / `fraction` modes; soft `max_output_frames` backstop.
- **`coverage.py`** — `fill_gaps`: walks all decoded frames and force‑keeps the best frame in any window that would exceed `max_gap_seconds`; tags them `coverage` in `keep_reasons`. `largest_gap_frames` reports the achieved density.
- **`scene.py`** — `detect_scene_changes`: optional histogram‑correlation scene/key‑frame marking on the raw extraction (available for callers that want scene‑aware keeps; not wired into the default `run_dvpe` flow).
- **`output.py`** — `write_outputs`: writes `metadata.json` + `processing_summary.json` (with a score‑ranked best‑frame manifest); frames are copied unmodified by the orchestrator.

It reuses the shared **validator** (`services/validator.py`, with a lenient config), the shared
`FrameRecord` **schema**, and the shared image‑metric / probe / IO **utilities**; everything else
is DVPE‑specific.

---

## 5. Shared data model · `models/schema.py`

Every stage passes the same shapes; no module invents its own dict.

- **`FrameRecord`** — the object every stage reads/writes: id/index/path/timestamp; quality (`sharpness/brightness/contrast/noise_sigma/overexposed/underexposed/dark_fraction/bright_fraction/passed_quality/quality_score`); dedup (`phash/is_duplicate/duplicate_of`); `scene_id`, `is_scene_change`, `keep_reasons` (keep/reject audit trail); scoring (`final_score/score_breakdown`); selection (`selected_as_best`).
- **`MissionMetadata`** — mission/video/drone/camera, resolution/fps/duration/codec/frame_count, GPS/altitude/heading/yaw/pitch/roll, capture time/flight speed/camera angle.
- **`BBox` / `Detection`** — geometry primitives retained for the shared `FrameRecord.detections` field; DVPE never populates them (no detection runs), so they stay empty.

---

## 6. Repository layout

```
dvie_pipeline/
├── venv/                             # Python 3.13 virtualenv
└── dvie_package/                     # <- Docker build context
    ├── PROJECT.md                    # this file
    ├── README.md                     # quickstart
    ├── DOCKER.md                     # container reference
    ├── Dockerfile                    # multi-stage, non-root, ffmpeg optional
    ├── docker-compose.yml            # dvpe / tower-grouping / tests / shell services
    ├── .dockerignore
    ├── requirements.txt
    ├── pytest.ini
    ├── videos/                       # input footage mount (-> /app/videos, read-only)
    ├── data/
    │   └── missions/                 # pipeline output lands here (per mission)
    ├── tests/
    │   └── test_dvpe_smoke.py        # DVPE end-to-end + coverage + all-mode
    └── video_processing/
        ├── config.py                 # shared validation/quality/duplicate/scene defaults
        ├── dvpe/                     # DVPE preprocessing engine
        │   ├── config.py             pipeline.py    metadata.py
        │   ├── extraction.py         quality_gate.py dedup.py
        │   ├── coverage.py           scoring.py      selection.py
        │   ├── scene.py              output.py
        ├── tower_grouping/           # OPT-IN post-processing: group best frames per tower
        │   ├── config.py             gps.py          towers.py
        │   ├── assign.py             grouping.py     __main__.py
        ├── models/
        │   └── schema.py             # shared dataclasses
        ├── services/
        │   └── validator.py          # shared Layer-1 validator (used with a lenient config)
        └── utils/                    # shared helpers
            ├── image_ops.py          # cv2/numpy metric helpers
            ├── io_helpers.py         # logging + JSON (utf-8)
            └── probe.py              # ffprobe-or-OpenCV video probe
```

---

## 7. Environment & installation

**Verified environment (this machine):**
- Python **3.13.5**, Windows 11
- opencv‑python, numpy, scikit‑image, imagehash, Pillow
- **ffmpeg/ffprobe: installed** (ffprobe **8.1.2**, Gyan build, on PATH). Real runs report `"probe_source": "ffprobe"`. It remains *optional* — DVPE falls back to OpenCV automatically if it's absent.

**Install:**
```powershell
pip install -r requirements.txt
```
`ffmpeg`/`ffprobe` gives the most accurate probe when on PATH; otherwise OpenCV is used
automatically (`winget install Gyan.FFmpeg` to add it).

**Docker (alternative — no local Python/OpenCV/ffmpeg needed):** `Dockerfile` +
`docker-compose.yml` live in `dvie_package/` (build context = `dvie_package/`, so the host
`venv/` is never in an image). Multi-stage build: deps compile into `/opt/venv` in a builder
stage, the runtime stage carries only that venv plus `libgl1`/`libglib2.0-0` (opencv‑python's
shared objects) and ffmpeg (`--build-arg INSTALL_FFMPEG=false` to drop it and fall back to the
OpenCV probe). Runs non‑root as uid 1000, base pinned to `python:3.13-slim-bookworm`.

```powershell
# From dvie_package: footage in ./videos, output in ./data/missions/
docker compose build
docker compose run --rm dvpe --video /app/videos/r.mp4 --mission-id Mission_001
docker compose run --rm tower-grouping --mission-id Mission_001 `
    --towers /app/configs/towers.example.json --gps /app/videos/track.srt
docker compose run --rm tests
```

Mounts: `./videos → /app/videos` (ro), `./data → /app/data` (rw), `./configs → /app/configs`
(ro). Full reference in `DOCKER.md`.

---

## 8. How to run

```powershell
# From dvie_package:
python -m video_processing.dvpe.pipeline --video path\to\video.mp4 --mission-id Mission_001

# Options:
#   --mode {auto,all,count,fraction}   selection strategy (default: auto = density cap)
#   --fps 2.0                          auto mode: max kept frames per second of video
#   --target-count 50                  for --mode count
#   --target-fraction 0.30             for --mode fraction
#   --max-gap-seconds 1.0              coverage safety rule
#   --max-output 2000                  soft cap on output frames
#   --keep-frames                      keep the intermediate decoded-frames cache
```

Programmatic:
```python
from video_processing.dvpe.config import DVPEConfig
from video_processing.dvpe.pipeline import run_dvpe

summary = run_dvpe("video.mp4", "Mission_001", DVPEConfig())
```

### Output

```
missions/Mission_001/
├── best_frames/               # selected frames, copied UNMODIFIED (native resolution)
│   └── frameNNNNN.jpg
├── metadata.json              # video metadata
└── processing_summary.json    # counts, per-stage timings, ranked best-frame manifest
```

`processing_summary.json` keys: `mission_id, accepted, video_name, original_frame_count,
frames_decoded, frames_rejected_by_quality, quality_rejections{reason:count},
frames_removed_as_duplicates, coverage_forced_frames, frames_kept, largest_gap_frames,
largest_gap_seconds, selection_mode, timings_sec{stage:sec}, processing_time_sec, best_frames[…]`.
Each `best_frames[]` entry: `frame_id, frame_index, timestamp_sec, coverage_forced, score,
score_rank, score_breakdown, file, written`.

The intermediate `frames/` cache is deleted after the run unless `--keep-frames` is set.

---

## 9. Configuration reference · `dvpe/config.py` (`DVPEConfig`)

| Group | Key | Default | Meaning |
|-------|-----|---------|---------|
| Validation | (lenient) | any decodable, `min_duration_sec=0.2` | no fps/resolution/codec gate |
| Decode | `max_decode_frames` | 3000 | stride cap for very long videos |
| | `jpg_quality` | 95 | written frame quality |
| Quality reject | `extreme_blur_floor` | 12.0 | noise‑normalized var‑of‑Laplacian floor |
| | `black_brightness` / `white_brightness` | 12.0 / 243.0 | all‑black / all‑white cutoffs |
| | `dark_fraction_max` / `bright_fraction_max` | 0.985 / 0.985 | clipped‑pixel fractions |
| | `min_contrast` | 5.0 | no‑visible‑detail floor |
| | `noise_penalty_k` | 0.05 | noise‑normalize sharpness |
| Duplicate | `phash_hamming_threshold` | 10 | pHash prefilter distance |
| | `ssim_threshold` | 0.92 | confirm near‑identical |
| Coverage | `max_gap_seconds` | 1.0 | no gap between kept frames may exceed this |
| Scoring | `weights` | sharpness .40 / contrast .20 / brightness .15 / noise .15 / exposure .10 | image‑quality only |
| Selection | `mode` | `auto` | auto (density cap) / all / count / fraction |
| | `keep_fps` | 2.0 | auto mode: max kept frames per second |
| | `target_count` / `target_fraction` | None / 0.30 | for count / fraction modes |
| | `min_frame_gap` / `min_output_frames` | 1 / 1 | |
| | `max_output_frames` | 2000 | soft cap backstop |
| Engine | `keep_extracted_frames` | False | keep the intermediate frame cache |

Shared defaults (validation/quality/duplicate/scene) live in `video_processing/config.py`.

---

## 10. Verified results — real footage (this machine)

| Mission | Video | Decoded | Rejected (quality) | Dups removed | **Kept** | Largest gap | Runtime |
|---------|-------|---------|--------------------|--------------|----------|-------------|---------|
| `Mission_1` | `r.mp4` (478×850 portrait, h264, 12.9s, 30fps) | 389 | 43 (extreme_blur) | 23 | **26** | 0.965s | 27.95s |
| `Mission_06` | `w.mp4` (240 frames) | 240 | 1 (extreme_blur) | 13 | **18** | 0.973s | 9.65s |

Both kept a duration‑proportionate set (auto/density‑cap, `keep_fps=2`), honored the ≤1s
coverage rule, produced a flat `best_frames/` package with native‑resolution copies, and needed
no coverage force‑keeps. Decode dominates runtime; dedup/scoring/selection/coverage are ~0s
thanks to inline metrics. Probe source: **ffprobe**.

---

## 11. Known limitations

- No semantic understanding — selection is image‑quality + temporal‑coverage only; it does not know what's *in* a frame (that's by design — detection belongs downstream).
- On extremely long/large videos the decode stride (`max_decode_frames=3000`) subsamples input, so the effective candidate set is bounded.
- Duplicate/coverage thresholds are tuned for typical drone clips; unusual footage may need `--fps` / `--max-gap-seconds` tuning.

---

## 12. Testing

```powershell
# From dvie_package:
python -m pytest tests/ -v
```
Current suite (**22 passing**):
- `test_dvpe_smoke.py` (3) — end‑to‑end density‑cap path, the coverage safety rule, and `all` mode.
- `test_dvpe_inspect.py` (4) — inspect‑mode composite selection / adaptive window.
- `test_tower_grouping.py` (15) — Haversine + nearest tower; the hysteresis switching rule
  (isolated flips are noise, sustained runs switch and reassign the run, no‑GPS/too‑far
  frames unassigned); the SRT/JSON/CSV GPS parsers + timestamp interpolation; and an
  end‑to‑end grouping run (copy and `--move`) on a temp mission fixture.

*Recommended next tests:* DVPE unit tests for the quality‑reject reasons and the density‑cap math.

---

## 13. How to extend

- **DVPE tuning:** all knobs in `dvpe/config.py`. Change retention via `--mode`/`--fps`, coverage via `--max-gap-seconds`.
- **Feed downstream:** point any inspection/detection stage at the `best_frames/` output.
- **Serving:** wrap `run_dvpe` in a FastAPI job endpoint + a small upload UI for non‑CLI users.

---

## 14. Tower‑wise grouping — optional post‑processing stage

DVPE itself stays preprocessing‑only (no GPS, no geometry). Organising the best frames
**per transmission tower** is a separate, **opt‑in** stage in
`video_processing/tower_grouping/` that runs *after* a mission is processed and never
touches the engine. It consumes the finished mission plus two external inputs.

**Inputs**
- **Tower database** (JSON) — fixed APTRANSCO tower coordinates, changes only when towers
  are added: `[{"tower_id":"T001","lat":17.734820,"lon":83.281210}, …]` (lat/latitude,
  lon/longitude aliases accepted). See `configs/towers.example.json`.
- **Per‑frame GPS** — auto‑detected by extension: a DJI **`.srt`** sidecar
  (`[latitude: …] [longitude: …]` or `GPS(lon,lat,alt)`, cue start time = fix time), or a
  **`.json`** / **`.csv`** log keyed by timestamp/frame (aliases: lat/latitude,
  lon/lng/longitude, timestamp/time/t, frame/file). *(DVPE captures no GPS, so this is
  supplied separately — the timestamps line up because both use seconds from video start.)*

**How it works** (spec Steps 1–7)
1. Read the mission's `processing_summary.json` best‑frame manifest.
2. Resolve each frame's GPS from the track: exact `frame_id` match wins, else linear
   interpolation between the two bracketing fixes (or nearest‑in‑time with `--no-interpolate`).
3. Haversine distance from the frame's GPS to every tower → nearest tower (raw decision).
4. **Hysteresis switching:** keep a *current tower*; a different tower must be nearest for
   `switch_patience` consecutive frames before the assignment switches (and the confirming
   run is then assigned to the new tower). Isolated flips stay with the current tower as GPS
   noise. Frames with no reliable GPS — or beyond `--max-distance` from every tower — are left
   unassigned and never disturb the current/candidate state.
5. Lay out `Tower_NNN/` folders (number parsed from the tower id: `T001 → Tower_001`), copy
   (default, non‑destructive) or `--move` the frames in, and write per‑tower `metadata.json`
   (`tower_id, frames, start_time, end_time, best_frame`, plus `tower_lat/lon`,
   `mean/max_distance_m`, `frame_files`). A `tower_grouping_summary.json` records the overall
   audit trail (source kinds, config, switches, assigned/unassigned counts).

**Run**
```powershell
python -m video_processing.tower_grouping --mission-id Mission_001 `
    --towers configs\towers.example.json --gps path\to\track.srt
# Options: --mission-dir DIR | --data-root DIR | --config FILE | --switch-patience N
#          --max-distance M | --no-interpolate | --gps-max-gap S | --move
```
Programmatic: `run_tower_grouping(towers_path, gps_path, mission_id=…, config=…)` in
`tower_grouping/grouping.py`. All knobs live in `tower_grouping/config.py`
(`TowerGroupingConfig`, JSON‑overridable — `configs/tower_grouping.example.json`).

**Output**
```
missions/Mission_001/
├── best_frames/                 # left intact (copy mode)
├── Tower_001/  frameNNNNN.jpg …  metadata.json
├── Tower_002/  …
└── tower_grouping_summary.json
```

---

## 15. Changelog

### 2026‑07‑27 — Docker packaging
1. Added `Dockerfile` (multi‑stage `python:3.13-slim-bookworm`, deps in `/opt/venv`, runtime
   carries only the venv + `libgl1`/`libglib2.0-0` + optional ffmpeg via
   `INSTALL_FFMPEG`, non‑root uid 1000, import smoke check at build time),
   `docker-compose.yml` (`dvpe` / `tower-grouping` / `tests` / `shell`), `.dockerignore`
   (excludes the host `venv/`, `data/`, caches, media) and `DOCKER.md`. Engine code
   unchanged — the container just mounts `./videos`, `./data`, `./configs`.

### 2026‑07‑20 — Tower‑wise grouping (opt‑in post‑processing)
1. Added `video_processing/tower_grouping/` — an opt‑in stage that groups a processed
   mission's best frames per tower from an external tower DB + per‑frame GPS
   (`.srt`/`.json`/`.csv`), using Haversine nearest‑tower + a hysteresis switching rule to
   absorb GPS noise. Writes `Tower_NNN/` folders + per‑tower `metadata.json` +
   `tower_grouping_summary.json`. CLI: `python -m video_processing.tower_grouping`. DVPE
   itself is unchanged and stays preprocessing‑only. Tests in `tests/test_tower_grouping.py`.

### 2026‑07‑20 — DVIE removed; repo is now DVPE‑only
1. **Removed the DVIE full‑intelligence engine** — `api/pipeline.py`, the DVIE‑only services (`detector`, `tracker`, `enhancer`, `extractor`, `scorer`, `selector`, `dataset_builder`, `metadata`, `quality`, `duplicate`, `scene`), the YOLO weights (`models/best.pt`), and the DVIE tests.
2. **Trimmed shared config** (`config.py`) to only the validation/quality/duplicate/scene defaults DVPE uses; removed detection/tracking/scoring/selection/enhancement/extraction config classes.
3. **Trimmed schema** — removed the DVIE‑only `TowerTrack`; kept `FrameRecord`/`MissionMetadata`/`BBox`/`Detection`.
4. **Dropped `ultralytics`** from `requirements.txt` (YOLO was DVIE‑only).

### Prior — DVPE engine
- Decode‑every‑frame, reject‑only‑unusable, remove‑only‑near‑identical, density‑capped selection, coverage safety rule, flat output. Verified on real footage (`Mission_1`, `Mission_06`).
- `FrameRecord` gained `dark_fraction`, `bright_fraction`, `is_scene_change`, `keep_reasons` for DVPE's keep/reject audit trail.
- ffmpeg/ffprobe installed (8.1.2); OpenCV probe fallback retained.
