"""
Tower-wise grouping — an OPT-IN post-processing stage that runs *after* DVPE.

DVPE is preprocessing-only: it produces a flat ``best_frames/`` package with no
detection, no geometry, and no GPS. This package takes that finished mission output and
organises the frames tower-by-tower using externally-supplied data:

    APTRANSCO tower database (JSON)  +  per-frame drone GPS (DJI .SRT / JSON / CSV)
                                    |
                                    v
        for each best frame:  look up its GPS -> Haversine distance to every tower
                              -> nearest tower -> hysteresis switching (ignore GPS noise)
                                    |
                                    v
        Mission_XX/
            Tower_001/  frameNNNNN.jpg ...  metadata.json
            Tower_002/  ...
            tower_grouping_summary.json

It is intentionally decoupled from the engine: DVPE stays model-free and geometry-free,
and this stage never touches the preprocessing logic — it only consumes the output.

Entry point: ``python -m video_processing.tower_grouping``
"""
