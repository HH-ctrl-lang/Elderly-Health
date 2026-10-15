"""Command line entry points with explicit inputs and isolated experiment outputs."""

import argparse
import hashlib
import json
import logging
import pickle
import platform
import sys
from dataclasses import fields, replace
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn

from .config import TrainConfig
from .data import (
    ENVIRONMENT,
    FEATURES,
    held_out_split,
    prepare_splits,
    random_split,
    read_raw,
    save_splits,
)
from .graph import load_graph_prior

PROJECT = Path(__file__).resolve().parents[2]


def write_json(path, value):
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )


def load_config(args):
    values = {}
    if args.config:
        values = json.loads(Path(args.config).read_text(encoding="utf-8"))
        unknown = set(values) - {f.name for f in fields(TrainConfig)}
        if unknown:
            raise ValueError(f"Unknown configuration keys: {sorted(unknown)}")
    for name in ("model", "seed", "epochs", "device", "threads", "probe_epochs", "irm_mode"):
        value = getattr(args, name, None)
        if value is not None:
            values[name] = value
    return TrainConfig(**values)


def provenance(args, config):
    import torch

    path = Path(args.input)
    return {
        "config": config.to_dict(),
        "input_path": str(path.resolve()),
        "input_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "feature_names": list(FEATURES),
        "label_encoding": "0=no impairment;1=impairment",
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "sklearn": sklearn.__version__,
        "graph_path": str(Path(args.graph).resolve()) if args.graph else None,
        "graph_sha256": hashlib.sha256(Path(args.graph).read_bytes()).hexdigest()
        if args.graph
        else None,
    }


def run_experiment(args, config, frames, output):
    # Optional dependency: preprocessing and --help do not require importing torch.
    import torch

    from .evaluation import evaluate, export_predictions, save_curves
    from .training import fit, get_device, predict

    prior = None
    if args.graph and config.model != "baseline" and config.lambda_graph > 0:
        prior = load_graph_prior(args.graph, "cognitive_impairment", FEATURES)
    output.mkdir(parents=True, exist_ok=False)
    splits, preprocessor = prepare_splits(frames)
    write_json(output / "config.json", provenance(args, config))
    save_splits(output / "splits.npz", splits)
    with (output / "preprocessor.pkl").open("wb") as stream:
        pickle.dump(preprocessor, stream)
    model, training = fit(splits[0], splits[1], config, prior)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": config.to_dict(),
            "feature_names": list(FEATURES),
            "env_map": training["env_map"],
            "best_epoch": training["best_epoch"],
        },
        output / "best_model.pt",
    )
    write_json(output / "training.json", training)
    result = evaluate(model, splits, config, prior)
    write_json(output / "metrics.json", result)
    export_predictions(model, splits, config, output)
    save_curves(splits[2].y, predict(model, splits[2].x, get_device(config.device)), output)
    print(f"Saved {output}; test metrics: {result['main']}", flush=True)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("preprocess", "train", "loeo"):
        sub = subparsers.add_parser(command)
        sub.add_argument("--input", type=Path, default=PROJECT / "data/data_mission2.csv")
        sub.add_argument("--encoding", default="gbk")
        sub.add_argument("--output", type=Path, required=True)
        sub.add_argument("--seed", type=int, default=None)
        if command != "preprocess":
            sub.add_argument("--config", type=Path)
            sub.add_argument("--model", choices=["baseline", "v4", "v5"], default=None)
            sub.add_argument("--epochs", type=int, default=None)
            sub.add_argument("--device", choices=["cpu", "cuda", "auto"], default=None)
            sub.add_argument("--threads", type=int, default=None)
            sub.add_argument("--probe-epochs", type=int, default=None)
            sub.add_argument("--irm-mode", choices=["none", "standard", "v11"], default=None)
            # None explicitly means no graph. Missing paths fail instead of silently disabling it.
            sub.add_argument("--graph", type=Path, default=None)
        if command == "loeo":
            sub.add_argument("--held-out", help="One site code; omit to evaluate every site")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    frame = read_raw(args.input, args.encoding)
    if args.command == "preprocess":
        args.output.mkdir(parents=True, exist_ok=False)
        seed = 42 if args.seed is None else args.seed
        splits, preprocessor = prepare_splits(random_split(frame, seed))
        save_splits(args.output / "splits.npz", splits)
        with (args.output / "preprocessor.pkl").open("wb") as stream:
            pickle.dump(preprocessor, stream)
        print(f"Saved train-only preprocessing to {args.output}")
        return
    config = load_config(args)
    if args.command == "train":
        run_experiment(args, config, random_split(frame, config.seed), args.output)
        return
    # Every outer fold starts from raw data and learns its own imputer and scaler.
    sites = sorted(frame[ENVIRONMENT].unique()) if args.held_out is None else [args.held_out]
    args.output.mkdir(parents=True, exist_ok=False)
    rows = []
    for site in sites:
        frames = held_out_split(frame, site, config.seed)
        for model_name in (
            (config.model, "baseline") if config.model != "baseline" else ("baseline",)
        ):
            fold_config = replace(config, model=model_name)
            # Original model and baseline see exactly the same rows in each fold.
            result = run_experiment(args, fold_config, frames, args.output / str(site) / model_name)
            rows.append({"site": str(site), "model": model_name, **result["main"]})
            pd.DataFrame(rows).to_csv(args.output / "summary.csv", index=False)
    print("LOEO complete; use summary.csv for paired per-site comparisons.")


if __name__ == "__main__":
    main(sys.argv[1:])
