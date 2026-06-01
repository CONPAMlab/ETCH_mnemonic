from __future__ import annotations

import argparse
import json
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="etch", description="Ego-NVS extraction toolkit")
    sub = parser.add_subparsers(dest="command", required=True)

    extract = sub.add_parser("extract", help="Run D.4.1 natural vision statistics extraction")
    extract.add_argument("--config", required=True, help="Path to a YAML config")

    validate = sub.add_parser("validate", help="Validate an extraction run directory")
    validate.add_argument("--run-dir", required=True)

    summarize = sub.add_parser("summarize", help="Summarize an extraction run directory")
    summarize.add_argument("--run-dir", required=True)

    predictive = sub.add_parser("model-predictive", help="Build D.4.2 predictive-coding proxy targets")
    predictive.add_argument("--features", required=True, help="Run directory or object parquet shard")
    predictive.add_argument("--output", default=None, help="Optional output directory")

    args = parser.parse_args(argv)
    if args.command == "extract":
        from .config import load_config
        from .extract import run_extraction

        result = {"run_dir": str(run_extraction(load_config(args.config)))}
    elif args.command == "validate":
        from .validate import validate_run

        result = validate_run(Path(args.run_dir))
    elif args.command == "summarize":
        from .summarize import summarize_run

        result = summarize_run(Path(args.run_dir))
    elif args.command == "model-predictive":
        from .predictive import build_predictive_scaffold

        result = build_predictive_scaffold(Path(args.features), Path(args.output) if args.output else None)
    else:
        parser.error(f"Unknown command: {args.command}")

    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
