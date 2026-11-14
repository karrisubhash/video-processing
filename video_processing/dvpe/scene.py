"""
Scene-Change Detection + Key-Frame Marking (DVPE).

Runs immediately after frame extraction, on ALL extracted frames (before quality
filtering and duplicate removal), per the flow:

    Frame Extraction -> Scene-Change Detection -> Mark Scene-Change Frame (key frame)
    -> Duplicate Removal -> Best Frame Selection

A scene change is a drop in grayscale-histogram correlation between consecutive
extracted frames below ``SceneConfig.hist_corr_threshold`` (camera turning, subject
entering/leaving, zoom, altitude change). The frame that opens each scene — including
the very first frame — is marked ``is_scene_change=True`` and becomes a key-frame
candidate that downstream stages always keep.

Unlike ``services/scene.py`` (DVIE Layer 6, which only labels quality-passed,
non-duplicate frames), this operates on the raw extraction so key frames are identified
before anything can drop them.
"""
from __future__ import annotations
import cv2

from video_processing.config import SceneConfig, DEFAULT_CONFIG
from video_processing.utils.image_ops import to_gray, hist_correlation
from video_processing.utils.io_helpers import get_logger

logger = get_logger("dvpe.scene")


def detect_scene_changes(records: list, config: SceneConfig = None) -> list:
    """Assigns ``scene_id`` and marks ``is_scene_change`` on each record in place.
    Records are processed in frame-index (extraction) order.
    """
    config = config or DEFAULT_CONFIG.scene
    ordered = sorted(records, key=lambda r: r.frame_index)

    scene_num = 0
    prev_gray = None
    for record in ordered:
        img = cv2.imread(record.path)
        gray = to_gray(img) if img is not None else None

        if prev_gray is None or gray is None:
            # First frame (or unreadable) opens a new scene.
            scene_num += 1
            record.is_scene_change = True
        else:
            corr = hist_correlation(prev_gray, gray)
            if corr < config.hist_corr_threshold:
                scene_num += 1
                record.is_scene_change = True
            else:
                record.is_scene_change = False

        record.scene_id = f"scene{scene_num:03d}"
        if gray is not None:
            prev_gray = gray

    key_frames = sum(1 for r in ordered if r.is_scene_change)
    logger.info(
        f"Scene-change detection: {scene_num} scene(s), {key_frames} key frame(s) marked "
        f"across {len(ordered)} extracted frame(s)"
    )
    return records
