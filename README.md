# APTRANSCO Drone Video Preprocessing Engine (DVPE)

Turn a raw drone inspection video into a small, clean set of the best usable,
non-duplicate frames — plus metadata and a processing summary — so downstream
AI/analytics run faster and more accurately.

**Preprocessing only.** DVPE deliberately contains **no** tower detection,
tracking, YOLO, SAM, or image enhancement. It keeps every genuinely useful,
distinct frame and removes only what is clearly worthless or redundant, so the
output scales with the video's real content. Whatever inspection system consumes
`best_frames/` does the detection.

## Pipeline

```
Drone Video → Validation (lenient) → Metadata → Decode EVERY frame →
Quality Analysis (reject ONLY unusable) → Near-Duplicate Removal (only near-identical) →
Frame Scoring (image quality only) → Best-Frame Selection (density cap) →
Coverage Check (no gap > max_gap_seconds) → Output (flat package)
```

**Design principle:** no single feature is ever trusted alone — blur **+**
brightness **+** contrast gate quality; perceptual-hash **+** SSIM confirm
duplicates; a time-interval coverage rule guarantees temporal density.

## Setup

```bash
pip install -r requirements.txt
```

**`ffmpeg`/`ffprobe` is optional.** When present on PATH it's used for the most
accurate container/codec/duration probe. When it's missing, DVPE automatically
falls back to OpenCV's probe (`video_processing/utils/probe.py`), so everything
runs on a stock machine — no install required. To add it anyway:
`apt install ffmpeg` / `brew install ffmpeg` / `winget install Gyan.FFmpeg`.

### Or use Docker

No Python, OpenCV or ffmpeg install needed — the image ships all three:

```powershell
# From dvie_package: put footage in ./videos, output lands in ./data/missions/
docker compose build
docker compose run --rm dvpe --video /app/videos/r.mp4 --mission-id Mission_001
```

See [DOCKER.md](DOCKER.md) for the mount layout, tower-grouping/test services,
build args and Linux permission notes.

## Run

Windows (PowerShell):

```powershell
# From dvie_package:
python -m video_processing.dvpe.pipeline --video path\to\video.mp4 --mission-id Mission_001
```

macOS/Linux:

```bash
python -m video_processing.dvpe.pipeline --video path/to/video.mp4 --mission-id Mission_001
```

Options:

```
--mode {auto,all,count,fraction}   selection strategy (default: auto = density cap)
--fps 2.0                          auto mode: max kept frames per second of video
--target-count 50                  for --mode count
--target-fraction 0.30             for --mode fraction
--max-gap-seconds 1.0              coverage safety rule (max seconds between kept frames)
--max-output 2000                  soft cap on output frames
--keep-frames                      keep the intermediate decoded-frames cache
```

No model or weights are needed.

## Output

Output lands in `data/missions/<mission_id>/`:

```
missions/Mission_001/
├── best_frames/               # selected frames, copied UNMODIFIED (native resolution)
│   └── frameNNNNN.jpg
├── metadata.json              # video metadata
└── processing_summary.json    # counts + per-stage timings + ranked best-frame manifest
```

The intermediate `frames/` cache is deleted after the run unless `--keep-frames` is set.

## Using it programmatically

```python
from video_processing.dvpe.config import DVPEConfig
from video_processing.dvpe.pipeline import run_dvpe

summary = run_dvpe("video.mp4", "Mission_001", DVPEConfig())
```

## Config

All DVPE knobs live in `video_processing/dvpe/config.py` (`DVPEConfig`) — nothing
is hardcoded inside the modules, so tuning retention (`--mode`/`--fps`) or coverage
(`--max-gap-seconds`) for your actual drone/camera doesn't require touching pipeline
logic. Shared validation/quality/duplicate/scene defaults live in
`video_processing/config.py`.

## Tower-wise grouping (optional, runs after DVPE)

DVPE stays preprocessing-only. If you also want the best frames organised **per
transmission tower**, an opt-in post-processing stage does that using two external
inputs — the APTRANSCO tower database and the drone's per-frame GPS:

```powershell
# From dvie_package, after a mission has been processed:
python -m video_processing.tower_grouping `
    --mission-id Mission_001 `
    --towers configs\towers.example.json `
    --gps    path\to\track.srt        # or .json / .csv
```

It reads the mission's `processing_summary.json`, resolves each frame's GPS from the
track, computes the Haversine distance to every tower, picks the nearest, and applies a
**hysteresis switching rule** so short GPS glitches don't bounce frames between towers.
The result sits alongside `best_frames/`:

```
missions/Mission_001/
├── best_frames/                 # left intact (copy mode; use --move to relocate)
├── Tower_001/
│   ├── frameNNNNN.jpg ...
│   └── metadata.json            # {tower_id, frames, start_time, end_time, best_frame, ...}
├── Tower_002/ ...
└── tower_grouping_summary.json  # overall audit trail
```

**GPS source** is auto-detected by extension: a DJI `.srt` sidecar
(`[latitude: …] [longitude: …]` or `GPS(lon,lat,alt)`), or a `.json` / `.csv` log keyed by
timestamp/frame (lat/latitude, lon/lng/longitude, timestamp/time aliases all accepted).

Key options: `--switch-patience N` (consecutive frames a new tower must win before
switching, default 3), `--max-distance M` (leave frames farther than M metres from every
tower unassigned), `--no-interpolate`, `--gps-max-gap S`, `--move`, and `--config FILE`
(see `configs/tower_grouping.example.json`).

## Tests

```powershell
# From dvie_package:
python -m pytest tests/ -v
```

A synthetic-video smoke test lives in `tests/test_dvpe_smoke.py` — it covers the
density-cap end-to-end path, the coverage safety rule, and `all` mode. Useful for
CI before you have real footage checked in.
