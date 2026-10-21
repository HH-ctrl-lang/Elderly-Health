"""Run a DomainBed VREx tabular adaptation on the project's existing splits.

Example (from repository root):
python project/run_external_vrex.py --mode loro --output outputs/external-vrex-region-loro-42
"""

import argparse
import hashlib
import json
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from elderly_health.data import (  # noqa: E402
    ENVIRONMENT,
    FEATURES,
    REGION,
    attach_regions,
    held_out_split,
    prepare_splits,
    random_split,
    read_raw,
)
from elderly_health.evaluation import metrics  # noqa: E402
from elderly_health.training import get_device, predict  # noqa: E402
from elderly_health.vrex_adapter import (  # noqa: E402
    DOMAINBED_SHA,
    VRExSettings,
    fit_vrex,
)


def write_json(path, value):
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )


def run_fold(frames, output, settings, environment_granularity, source_info):
    output.mkdir(parents=True, exist_ok=False)
    env_column = REGION if environment_granularity == "region" else ENVIRONMENT
    (train, val, test), preprocessing = prepare_splits(
        frames, environment_column=env_column
    )
    with (output / "preprocessor.pkl").open("wb") as stream:
        pickle.dump(preprocessing, stream)
    write_json(output / "config.json", {
        "algorithm": "VREx (binary tabular adaptation)",
        "source_algorithm": "DomainBed/VREx",
        "source_commit": DOMAINBED_SHA,
        "domainbed_default_vrex_lambda": 10.0,
        "domainbed_default_anneal_steps": 500,
        "domainbed_license": "MIT",
        "config": settings.to_dict(),
        "environment_granularity": environment_granularity,
        **source_info,
    })
    model, training = fit_vrex(train, val, settings)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "algorithm": "tabular_vrex",
            "features": list(FEATURES),
            "settings": settings.to_dict(),
            "best_epoch": training["best_epoch"],
        },
        output / "best_model.pt",
    )
    write_json(output / "training.json", training)
    prob = predict(model, test.x, get_device(settings.device))
    result = metrics(test.y, prob)
    write_json(output / "metrics.json", {"main": result})
    pd.DataFrame({
        "sample_id": test.sample_ids,
        "site_code": test.site_env,
        "training_environment_granularity": environment_granularity,
        "cognitive_impairment": test.y.astype(np.int32),
        "y_prob": prob,
    }).to_csv(output / "predictions_test.csv", index=False)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "data/data_mission2.csv")
    parser.add_argument("--encoding", default="gbk")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=["random", "loro"], default="loro")
    parser.add_argument("--environment-granularity", choices=["site", "region"], default="region")
    parser.add_argument("--held-out", choices=["east", "central", "west"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--selection-start-epoch", type=int, default=91)
    parser.add_argument("--patience", type=int, default=80)
    parser.add_argument("--vrex-lambda", type=float, default=10.0)
    parser.add_argument("--penalty-anneal-steps", type=int, default=500)
    parser.add_argument("--per-environment-batch-size", type=int, default=32)
    parser.add_argument("--device", choices=["cpu", "auto", "cuda"], default="cpu")
    args = parser.parse_args(argv)
    if args.mode != "loro" and args.held_out is not None:
        parser.error("--held-out is only valid with --mode loro")
    settings = VRExSettings(
        seed=args.seed,
        device=args.device,
        epochs=args.epochs,
        selection_start_epoch=args.selection_start_epoch,
        patience=args.patience,
        vrex_lambda=args.vrex_lambda,
        penalty_anneal_steps=args.penalty_anneal_steps,
        per_environment_batch_size=args.per_environment_batch_size,
    )
    if args.output.exists():
        parser.error(f"Output already exists; refusing to overwrite: {args.output}")
    frame = read_raw(args.input, args.encoding)
    if args.mode == "loro" or args.environment_granularity == "region":
        frame = attach_regions(frame)
    source = {
        "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "holdout_mode": args.mode,
    }
    args.output.mkdir(parents=True, exist_ok=False)
    if args.mode == "random":
        result = run_fold(
            random_split(frame, args.seed),
            args.output / "random",
            settings, args.environment_granularity, source,
        )
        rows = [{"held_out": "random", "model": "tabular_vrex", **result}]
    else:
        areas = [args.held_out] if args.held_out else ["east", "central", "west"]
        rows = []
        for area in areas:
            frames = held_out_split(frame, area, args.seed, column=REGION)
            result = run_fold(
                frames,
                args.output / area,
                settings, args.environment_granularity, source,
            )
            rows.append({
                "held_out": area, "model": "tabular_vrex",
                "environment_granularity": args.environment_granularity, **result,
            })
            pd.DataFrame(rows).to_csv(args.output / "summary.csv", index=False)
            print(f"Completed {area}: AUC={result['auc']:.4f}, AP={result['auprc']:.4f}")
    pd.DataFrame(rows).to_csv(args.output / "summary.csv", index=False)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
