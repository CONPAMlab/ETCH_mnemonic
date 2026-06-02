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

    train = sub.add_parser("train-predictive", help="Train the D.4.2 object-centric predictive coding model")
    train.add_argument("--config", help="Optional predictive-model YAML config")
    train.add_argument("--features", help="Run directory or object parquet shard")
    train.add_argument("--output", help="Output directory for checkpoint and alignment table")
    train.add_argument("--sequence-length", type=int, default=8)
    train.add_argument("--prediction-horizon", type=int, default=1)
    train.add_argument("--hidden-dim", type=int, default=128)
    train.add_argument("--latent-dim", type=int, default=64)
    train.add_argument("--batch-size", type=int, default=64)
    train.add_argument("--epochs", type=int, default=5)
    train.add_argument("--learning-rate", type=float, default=1e-3)
    train.add_argument("--min-track-length", type=int, default=12)
    train.add_argument("--device", default="cpu")
    train.add_argument("--seed", type=int, default=123)

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
    elif args.command == "train-predictive":
        from .predictive_model import PredictiveTrainingConfig, train_predictive_model

        train_args = _load_predictive_train_args(args)
        result = train_predictive_model(
            Path(train_args["features"]),
            Path(train_args["output"]),
            PredictiveTrainingConfig(
                sequence_length=train_args["sequence_length"],
                prediction_horizon=train_args["prediction_horizon"],
                hidden_dim=train_args["hidden_dim"],
                latent_dim=train_args["latent_dim"],
                batch_size=train_args["batch_size"],
                epochs=train_args["epochs"],
                learning_rate=train_args["learning_rate"],
                min_track_length=train_args["min_track_length"],
                device=train_args["device"],
                seed=train_args["seed"],
            ),
        )
    else:
        parser.error(f"Unknown command: {args.command}")

    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def _load_predictive_train_args(args: argparse.Namespace) -> dict:
    values = {
        "features": args.features,
        "output": args.output,
        "sequence_length": args.sequence_length,
        "prediction_horizon": args.prediction_horizon,
        "hidden_dim": args.hidden_dim,
        "latent_dim": args.latent_dim,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "min_track_length": args.min_track_length,
        "device": args.device,
        "seed": args.seed,
    }
    if args.config:
        import yaml

        with Path(args.config).open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        training = raw.get("training", {})
        values.update(
            {
                "features": raw.get("features", values["features"]),
                "output": raw.get("output", values["output"]),
                "sequence_length": training.get("sequence_length", values["sequence_length"]),
                "prediction_horizon": training.get("prediction_horizon", values["prediction_horizon"]),
                "hidden_dim": training.get("hidden_dim", values["hidden_dim"]),
                "latent_dim": training.get("latent_dim", values["latent_dim"]),
                "batch_size": training.get("batch_size", values["batch_size"]),
                "epochs": training.get("epochs", values["epochs"]),
                "learning_rate": training.get("learning_rate", values["learning_rate"]),
                "min_track_length": training.get("min_track_length", values["min_track_length"]),
                "device": training.get("device", values["device"]),
                "seed": training.get("seed", values["seed"]),
            }
        )
    if not values["features"] or not values["output"]:
        raise SystemExit("train-predictive requires --features and --output, or --config with features/output")
    return values


if __name__ == "__main__":
    raise SystemExit(main())
