"""
Decode Every Frame (DVPE).

The maximal-retention model starts by decoding every frame and treating each as a
candidate — nothing is filtered out here. Quality metrics (sharpness, brightness, contrast,
noise, exposure) are computed inline while the frame is still in memory, so no frame is read
back from disk for analysis (a big speedup over a separate pass). Downstream stages remove
only what is clearly worthless (quality) or redundant (near-duplicate); a coverage rule
preserves temporal density.

On extremely long/large videos a stride (ceil(total / max_decode_frames)) bounds the
candidate count and disk usage; for normal clips the stride is 1 (truly every frame). Each
decoded frame is written to disk and returned as a FrameRecord (with metrics populated) in
timeline order.
"""
from __future__ import annotations
import math
from pathlib import Path

import cv2
import imagehash
from PIL import Image

from video_processing.dvpe.config import DecodeConfig, DEFAULT_DVPE_CONFIG
from video_processing.models.schema import FrameRecord
from video_processing.utils.image_ops import (
    to_gray, variance_of_laplacian, mean_brightness, rms_contrast, estimate_noise_sigma, exposure_flags,
)
from video_processing.utils.io_helpers import get_logger

logger = get_logger("dvpe.decode")

_SSIM_THUMB_WIDTH = 128


def _measure(frame, record):
    """Compute quality metrics inline and stash a pHash + small grayscale thumbnail on the
    record so duplicate detection needs no disk re-reads (a big speedup)."""
    gray = to_gray(frame)
    exposure = exposure_flags(gray)
    record.sharpness = round(variance_of_laplacian(gray), 2)
    record.brightness = round(mean_brightness(gray), 2)
    record.contrast = round(rms_contrast(gray), 2)
    record.noise_sigma = round(estimate_noise_sigma(gray), 2)
    record.overexposed = exposure["overexposed"]
    record.underexposed = exposure["underexposed"]
    record.dark_fraction = round(exposure["dark_fraction"], 4)
    record.bright_fraction = round(exposure["bright_fraction"], 4)

    h, w = gray.shape
    if w > _SSIM_THUMB_WIDTH:
        scale = _SSIM_THUMB_WIDTH / float(w)
        small = cv2.resize(gray, (_SSIM_THUMB_WIDTH, max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
    else:
        small = gray
    record._gray_small = small                                  # for in-memory SSIM
    record._phash = imagehash.phash(Image.fromarray(gray))      # for near-dup prefilter
    record.phash = str(record._phash)


def decode_all_frames(video_path: str, output_dir: str, config: DecodeConfig = None) -> list:
    config = config or DEFAULT_DVPE_CONFIG.decode
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Could not open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

    stride = 1
    if config.max_decode_frames and total > config.max_decode_frames:
        stride = math.ceil(total / config.max_decode_frames)
        logger.info(f"Cost cap: {total} frames > {config.max_decode_frames} budget -> decode every {stride}th frame")

    write_params = [cv2.IMWRITE_JPEG_QUALITY, int(config.jpg_quality)]
    frame_idx = 0
    decoded = 0
    records: list = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % stride == 0:
            frame_id = f"frame{decoded:05d}"
            frame_path = out_dir / f"{frame_id}.jpg"
            cv2.imwrite(str(frame_path), frame, write_params)
            rec = FrameRecord(
                frame_id=frame_id,
                frame_index=frame_idx,
                path=str(frame_path),
                timestamp_sec=round(frame_idx / fps, 3),
            )
            _measure(frame, rec)          # metrics inline — no disk re-read
            records.append(rec)
            decoded += 1
        frame_idx += 1
    cap.release()
    logger.info(f"Decoded {decoded} candidate frame(s) from {frame_idx} total (stride={stride})")
    return records
