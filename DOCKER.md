# Running DVPE in Docker

Everything runs from `dvie_package/` — that is the build context (the sibling
`venv/` is deliberately outside it, so the host virtualenv never enters an image).

DVPE is a **batch CLI**, not a server, so the normal verb is
`docker compose run --rm …`, not `docker compose up`.

## Layout

| Host path | Container path | Mode | Purpose |
|---|---|---|---|
| `./videos/` | `/app/videos` | read-only | input drone footage (+ GPS sidecars) |
| `./data/` | `/app/data` | read-write | output — `data/missions/<mission_id>/` |
| `./configs/` | `/app/configs` | read-only | JSON config / tower DB overrides |

Output appears on the host under `data/missions/<mission_id>/` exactly as it does
for a native run — the container writes through the bind mount.

## Build

```powershell
docker compose build
```

Or without compose:

```powershell
docker build -t dvpe:latest .
```

**Build args**

- `INSTALL_FFMPEG` (default `true`) — installs `ffmpeg`/`ffprobe` so the metadata
  probe reports `"probe_source": "ffprobe"`. Set to `false` for a substantially
  smaller image; `utils/probe.py` falls back to OpenCV automatically.

  ```powershell
  docker build --build-arg INSTALL_FFMPEG=false -t dvpe:slim .
  ```

## Run the pipeline

Drop footage in `./videos/`, then everything after the service name goes straight
to the DVPE CLI:

```powershell
docker compose run --rm dvpe --video /app/videos/r.mp4 --mission-id Mission_001
```

All the usual flags work unchanged:

```powershell
docker compose run --rm dvpe `
    --video /app/videos/r.mp4 --mission-id Mission_002 `
    --mode inspect --max-gap-seconds 1.0 --max-output 500

docker compose run --rm dvpe `
    --video /app/videos/w.mp4 --mission-id Mission_003 `
    --config /app/configs/inspect.example.json
```

Plain `docker run` equivalent:

```powershell
docker run --rm `
    -v "${PWD}\data:/app/data" `
    -v "${PWD}\videos:/app/videos:ro" `
    dvpe:latest --video /app/videos/r.mp4 --mission-id Mission_001
```

## Tower-wise grouping

Runs after a mission exists. Keep the GPS track next to the footage in `./videos/`:

```powershell
docker compose run --rm tower-grouping `
    --mission-id Mission_001 `
    --towers /app/configs/towers.example.json `
    --gps    /app/videos/track.srt
```

## Tests

```powershell
docker compose run --rm tests
```

Fixtures are synthetic (no footage required) and the service mounts nothing, so a
test run cannot touch your missions.

## Shell

```powershell
docker compose run --rm shell
```

## Notes

- **Image design** — multi-stage: dependencies build into `/opt/venv` in a builder
  stage carrying `build-essential`; the runtime stage copies only the finished venv
  plus `libgl1`/`libglib2.0-0` (what `opencv-python` links against) and optional
  ffmpeg. No compilers or pip cache in the shipped image.
- **Base pinned to `bookworm`** rather than the floating `-slim` tag, because Debian
  trixie renames `libglib2.0-0` → `libglib2.0-0t64`.
- **Non-root** — runs as uid/gid `1000` (`dvpe`), and `/app` stays root-owned so the
  process cannot rewrite its own code. On Docker Desktop this needs no action. On a
  **Linux host**, bind mounts keep host ownership, so if `./data` isn't writable by
  uid 1000 either `chown -R 1000:1000 data` or run with
  `docker compose run --rm --user "$(id -u):$(id -g)" dvpe …`.
- **Long videos** — decode is the dominant cost and is CPU-bound. Give Docker
  Desktop enough CPU/RAM, or bound the work with `--max-output` / `--fps`.
- **Frame cache** — the intermediate `data/missions/<id>/frames/` cache is deleted at
  the end of a run unless `--keep-frames` is passed; on the bind mount it is real
  host disk, so watch space when keeping it.
