from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


def ensure_run_dirs(run_dir: Path) -> None:
    for subdir in [
        run_dir,
        run_dir / "features" / "object_tracks",
        run_dir / "features" / "frame_stats",
        run_dir / "qc",
    ]:
        subdir.mkdir(parents=True, exist_ok=True)


def assert_output_not_inside_dataset(output_root: Path, dataset_root: Path) -> None:
    out = output_root.expanduser().resolve()
    root = dataset_root.expanduser().resolve()
    if out == root or root in out.parents:
        raise ValueError(
            f"Refusing to write outputs under dataset root: output_root={out}, dataset_root={root}"
        )


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_table(path: Path, rows: list[dict[str, Any]] | pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    try:
        df.to_parquet(path, index=False)
    except ImportError as exc:
        raise RuntimeError(
            "Parquet output requires pyarrow or fastparquet. Install project requirements first."
        ) from exc


def read_tables(paths: list[Path]) -> pd.DataFrame:
    if not paths:
        return pd.DataFrame()
    return pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()

