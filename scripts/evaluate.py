"""Evaluate one checkpoint and export auditable logits/probabilities."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from hasnet.checkpoint import (
    load_checkpoint,
    load_model_weights,
    sha256_file,
    write_json,
)
from hasnet.config import build_model, config_hash, load_config
from hasnet.data import Perturbation
from hasnet.engine import evaluate, make_loader, save_predictions
from hasnet.losses import AsymmetricLoss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--weights", choices=["ema", "raw"], default="ema")
    parser.add_argument("--perturb-kind", choices=["clean", "missing", "blur"], default="clean")
    parser.add_argument("--perturb-view", choices=["ol", "sd"], default=None)
    parser.add_argument("--kernel-size", type=int, default=21)
    parser.add_argument("--sigma", type=float, default=5.0)
    parser.add_argument("--output-dir", default="outputs/evaluation")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.data_root is not None:
        config["data"]["root"] = args.data_root
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = load_checkpoint(args.checkpoint, "cpu")
    model = build_model(config, pretrained=False).to(device)
    load_model_weights(model, checkpoint, weights=args.weights)
    perturbation = Perturbation(
        kind=args.perturb_kind,
        view=args.perturb_view,
        kernel_size=args.kernel_size,
        sigma=args.sigma,
    )
    loader, _ = make_loader(
        config,
        args.split,
        context=None,
        train=False,
        perturbation=perturbation,
    )
    criterion = AsymmetricLoss(
        gamma_neg=float(config["loss"]["gamma_neg"]),
        gamma_pos=float(config["loss"]["gamma_pos"]),
        clip=float(config["loss"]["clip"]),
    )
    result = evaluate(
        model=model,
        loader=loader,
        criterion=criterion,
        device=device,
        config=config,
        description=f"{args.split}:{perturbation.name}",
    )
    output_dir = Path(args.output_dir)
    stem = f"{config['name']}_{args.split}_{perturbation.name}"
    metadata = {
        "experiment": config["name"],
        "split": args.split,
        "condition": perturbation.name,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "checkpoint_weights": args.weights,
        "config_sha256": config_hash(config),
    }
    save_predictions(result, output_dir / f"{stem}.npz", metadata)
    summary = {
        **metadata,
        "loss": result.loss,
        "metrics": result.metrics,
        "calibration": result.calibration,
        "num_samples": int(result.targets.shape[0]),
    }
    write_json(output_dir / f"{stem}.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
