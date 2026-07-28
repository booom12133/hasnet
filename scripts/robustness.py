"""Reproduce the five asymmetric one-view degradation conditions."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import yaml

from hasnet.checkpoint import load_checkpoint, load_model_weights, sha256_file, write_json
from hasnet.config import build_model, load_config
from hasnet.data import Perturbation
from hasnet.engine import evaluate, make_loader
from hasnet.losses import AsymmetricLoss


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--protocol", default="configs/paper/robustness.yaml")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--weights", choices=["ema", "raw"], default="ema")
    parser.add_argument("--output-dir", default="outputs/robustness")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.data_root is not None:
        config["data"]["root"] = args.data_root
    protocol = yaml.safe_load(Path(args.protocol).read_text(encoding="utf-8"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = load_checkpoint(args.checkpoint, "cpu")
    model = build_model(config, pretrained=False).to(device)
    load_model_weights(model, state, weights=args.weights)
    criterion = AsymmetricLoss(
        gamma_neg=float(config["loss"]["gamma_neg"]),
        gamma_pos=float(config["loss"]["gamma_pos"]),
        clip=float(config["loss"]["clip"]),
    )

    rows = []
    clean_map = None
    for condition in protocol["conditions"]:
        perturbation = Perturbation(
            kind=condition["kind"],
            view=condition.get("view"),
            kernel_size=int(condition.get("kernel_size", 21)),
            sigma=float(condition.get("sigma", 5.0)),
        )
        loader, _ = make_loader(
            config,
            protocol["split"],
            train=False,
            perturbation=perturbation,
        )
        result = evaluate(
            model=model,
            loader=loader,
            criterion=criterion,
            device=device,
            config=config,
            description=condition["name"],
        )
        current_map = float(result.metrics["mAP"])
        if condition["kind"] == "clean":
            clean_map = current_map
        if clean_map is None:
            raise ValueError("The clean condition must be listed first")
        rows.append(
            {
                "condition": condition["name"],
                "mAP": current_map,
                "retention_percent": 100.0 * current_map / clean_map,
                "kind": condition["kind"],
                "view": condition.get("view", ""),
                "kernel_size": condition.get("kernel_size", ""),
                "sigma": condition.get("sigma", ""),
            }
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{config['name']}_robustness.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    payload = {
        "experiment": config["name"],
        "split": protocol["split"],
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "weights": args.weights,
        "rows": rows,
    }
    write_json(output_dir / f"{config['name']}_robustness.json", payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
