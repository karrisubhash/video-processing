# syntax=docker/dockerfile:1
#
# APTRANSCO Drone Video Preprocessing Engine (DVPE)
#
# Multi-stage build: dependencies are compiled/installed into a virtualenv in the
# builder stage, then only the finished venv is carried into a slim runtime image
# (no compilers, no pip cache, no build headers).
#
#   docker build -t dvpe:latest .
#   docker run --rm -v "$PWD/data:/app/data" -v "$PWD/videos:/app/videos:ro" \
#       dvpe:latest --video /app/videos/r.mp4 --mission-id Mission_001
#
# Base is pinned to bookworm (not the floating -slim tag) so the runtime library
# names below (libgl1 / libglib2.0-0) stay valid across rebuilds.

# ---------------------------------------------------------------- builder ----
FROM python:3.13-slim-bookworm AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_ROOT_USER_ACTION=ignore

# Toolchain for any dependency without a prebuilt wheel for this platform
# (notably scikit-image / numpy on linux/arm64).
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copied alone so the (slow) dependency layer is cached until requirements change.
COPY requirements.txt ./
RUN pip install --upgrade pip setuptools wheel \
    && pip install -r requirements.txt

# ---------------------------------------------------------------- runtime ----
FROM python:3.13-slim-bookworm AS runtime

# ffprobe gives DVPE the most accurate container/codec/duration probe. It is
# genuinely optional — utils/probe.py falls back to OpenCV — so it can be dropped
# for a much smaller image with:  --build-arg INSTALL_FFMPEG=false
ARG INSTALL_FFMPEG=true

# libgl1 + libglib2.0-0 are the shared objects opencv-python links against.
# (Not needed if the project ever switches to opencv-python-headless.)
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
    && if [ "$INSTALL_FFMPEG" = "true" ]; then \
           apt-get install -y --no-install-recommends ffmpeg; \
       fi \
    && rm -rf /var/lib/apt/lists/*

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    OPENCV_LOG_LEVEL=ERROR

COPY --from=builder /opt/venv /opt/venv

WORKDIR /app

# Non-root by default. On a Linux host with bind mounts, the container UID must be
# able to write ./data — either run with `--user "$(id -u):$(id -g)"` or chown the
# host directory to 1000. Docker Desktop (Windows/macOS) needs neither.
RUN groupadd --gid 1000 dvpe \
    && useradd --uid 1000 --gid 1000 --create-home --shell /usr/sbin/nologin dvpe

# Source stays root-owned and world-readable: the engine can read and execute its own
# code but never rewrite it. Only the two data directories below are writable.
COPY pytest.ini README.md PROJECT.md ./
COPY configs/ ./configs/
COPY tests/ ./tests/
COPY video_processing/ ./video_processing/

# data/   -> pipeline output (data/missions/<mission_id>/); bind-mount a host dir here
# videos/ -> convention for the read-only input mount
RUN mkdir -p /app/data/missions /app/data/uploads /app/videos \
    && chown -R dvpe:dvpe /app/data /app/videos

USER dvpe

# Fails the build if the image cannot import the engine or reach OpenCV's binaries.
RUN python -c "from video_processing.dvpe.pipeline import run_dvpe; import cv2; print('cv2', cv2.__version__)"

ENTRYPOINT ["python", "-m", "video_processing.dvpe.pipeline"]
CMD ["--help"]
