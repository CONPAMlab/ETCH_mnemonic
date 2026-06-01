from __future__ import annotations

from pathlib import Path

import pandas as pd

from .io import read_json, write_json
from .schema import FRAME_STAT_COLUMNS, OBJECT_TRACK_COLUMNS, missing_columns


def validate_run(run_dir: str | Path) -> dict:
    run_dir = Path(run_dir)
    result = {
        "run_dir": str(run_dir),
        "status": "pass",
        "errors": [],
        "warnings": [],
        "n_object_shards": 0,
        "n_frame_shards": 0,
    }
    if not (run_dir / "metadata.json").exists():
        result["errors"].append("Missing metadata.json")
    if not (run_dir / "video_manifest.parquet").exists():
        result["errors"].append("Missing video_manifest.parquet")

    object_shards = sorted((run_dir / "features" / "object_tracks").glob("*.parquet"))
    frame_shards = sorted((run_dir / "features" / "frame_stats").glob("*.parquet"))
    result["n_object_shards"] = len(object_shards)
    result["n_frame_shards"] = len(frame_shards)
    if not object_shards:
        result["warnings"].append("No object track shards found")
    if not frame_shards:
        result["warnings"].append("No frame stat shards found")

    for path in object_shards[:10]:
        cols = list(pd.read_parquet(path).columns)
        miss = missing_columns(cols, OBJECT_TRACK_COLUMNS)
        if miss:
            result["errors"].append(f"{path.name} missing object columns: {miss}")
    for path in frame_shards[:10]:
        cols = list(pd.read_parquet(path).columns)
        miss = missing_columns(cols, FRAME_STAT_COLUMNS)
        if miss:
            result["errors"].append(f"{path.name} missing frame columns: {miss}")

    for qc_path in sorted((run_dir / "qc").glob("*.json")):
        qc = read_json(qc_path)
        if qc.get("status") != "complete":
            result["errors"].append(f"{qc_path.name} status is {qc.get('status')}")

    if result["errors"]:
        result["status"] = "fail"
    write_json(run_dir / "qc" / "run_validation.json", result)
    return result

