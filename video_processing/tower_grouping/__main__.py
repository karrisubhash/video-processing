"""
CLI for the tower-grouping post-processing stage. Run AFTER DVPE has produced a mission.

    # From dvie_package:
    python -m video_processing.tower_grouping \
        --mission-id Mission_04 \
        --towers path/to/towers.json \
        --gps path/to/track.srt          # or .json / .csv

Options:
    --mission-dir DIR        absolute mission dir (instead of --mission-id under data/missions)
    --data-root DIR          override the data root used to resolve --mission-id
    --config FILE            JSON config with knob overrides (see tower_grouping/config.py)
    --switch-patience N      consecutive frames a new tower must win before switching (default 3)
    --max-distance M         leave frames farther than M metres from every tower unassigned
    --no-interpolate         use the nearest-in-time GPS fix instead of interpolating
    --gps-max-gap S          max seconds a frame may be from its nearest fix (default 2.0)
    --move                   move frames into tower folders (default: copy, non-destructive)
"""
from __future__ import annotations
import argparse
import sys

from video_processing.tower_grouping.config import TowerGroupingConfig
from video_processing.tower_grouping.grouping import run_tower_grouping


def main():
    parser = argparse.ArgumentParser(
        description="Organise a DVPE mission's best frames into tower-wise folders.")
    parser.add_argument("--towers", required=True, help="Path to the tower database JSON")
    parser.add_argument("--gps", required=True, help="Per-frame GPS source (.srt / .json / .csv)")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--mission-id", help="Mission id under <data-root>/missions/")
    group.add_argument("--mission-dir", help="Absolute path to the mission directory")
    parser.add_argument("--data-root", default=None, help="Override data root for --mission-id")
    parser.add_argument("--config", default=None, help="JSON config with knob overrides")
    parser.add_argument("--switch-patience", type=int, default=None,
                        help="Consecutive frames a new tower must be nearest before switching")
    parser.add_argument("--max-distance", type=float, default=None,
                        help="Leave frames farther than this (metres) from every tower unassigned")
    parser.add_argument("--gps-max-gap", type=float, default=None,
                        help="Max seconds a frame may be from its nearest GPS fix")
    parser.add_argument("--no-interpolate", action="store_true",
                        help="Use the nearest-in-time GPS fix instead of interpolating")
    parser.add_argument("--move", action="store_true",
                        help="Move frames into tower folders (default: copy)")
    args = parser.parse_args()

    config = TowerGroupingConfig()
    if args.config is not None:
        from video_processing.tower_grouping.config import load_config
        config = load_config(args.config, base=config)
    if args.switch_patience is not None:
        config.switch_patience = args.switch_patience
    if args.max_distance is not None:
        config.max_assign_distance_m = args.max_distance
    if args.gps_max_gap is not None:
        config.gps_max_time_gap_sec = args.gps_max_gap
    if args.no_interpolate:
        config.interpolate_gps = False
    if args.move:
        config.move_frames = True

    try:
        run_tower_grouping(
            args.towers, args.gps,
            mission_id=args.mission_id, mission_dir=args.mission_dir,
            data_root=args.data_root, config=config,
        )
    except (FileNotFoundError, ValueError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
