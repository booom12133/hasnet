"""Fit temperature scaling without evaluating on its fitting observations."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from hasnet.calibration import TemperatureScaler
from hasnet.checkpoint import write_json
from hasnet.metrics import compute_calibration_metrics, compute_metrics


def load_archive(path: str | Path) -> dict:
    with np.load(path, allow_pickle=True) as archive:
        return {
            "logits": np.asarray(archive["logits"], dtype=np.float32),
            "targets": np.asarray(archive["targets"], dtype=np.int64),
            "indices": np.asarray(archive["indices"], dtype=np.int64),
            "metadata": archive["metadata"][0] if "metadata" in archive else {},
        }


def fit_temperature(logits: np.ndarray, targets: np.ndarray) -> tuple[TemperatureScaler, float]:
    scaler = TemperatureScaler()
    temperature = scaler.fit(torch.from_numpy(logits), torch.from_numpy(targets))
    return scaler, temperature


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", choices=["heldout", "crossfit"], default="heldout")
    parser.add_argument("--fit-predictions", required=True)
    parser.add_argument("--eval-predictions", default=None)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=3408)
    parser.add_argument("--num-bins", type=int, default=10)
    parser.add_argument("--output-dir", default="outputs/calibration")
    args = parser.parse_args()

    fit_data = load_archive(args.fit_predictions)
    temperatures: list[float] = []
    if args.protocol == "heldout":
        if not args.eval_predictions:
            raise ValueError("--eval-predictions is required for heldout calibration")
        if Path(args.fit_predictions).resolve() == Path(args.eval_predictions).resolve():
            raise ValueError("Fitting and evaluation archives must differ under heldout protocol")
        eval_data = load_archive(args.eval_predictions)
        scaler, temperature = fit_temperature(fit_data["logits"], fit_data["targets"])
        temperatures.append(temperature)
        with torch.no_grad():
            scaled_logits = scaler(torch.from_numpy(eval_data["logits"])).numpy()
    else:
        if args.eval_predictions and Path(args.fit_predictions).resolve() != Path(
            args.eval_predictions
        ).resolve():
            raise ValueError("Crossfit protocol operates on one archive; omit --eval-predictions")
        if args.folds < 2:
            raise ValueError("--folds must be at least 2")
        eval_data = fit_data
        rng = np.random.default_rng(args.seed)
        permutation = rng.permutation(len(fit_data["logits"]))
        fold_ids = np.empty(len(permutation), dtype=np.int64)
        fold_ids[permutation] = np.arange(len(permutation)) % args.folds
        scaled_logits = np.empty_like(fit_data["logits"])
        for fold in range(args.folds):
            fit_mask = fold_ids != fold
            eval_mask = ~fit_mask
            scaler, temperature = fit_temperature(
                fit_data["logits"][fit_mask], fit_data["targets"][fit_mask]
            )
            temperatures.append(temperature)
            with torch.no_grad():
                scaled_logits[eval_mask] = scaler(
                    torch.from_numpy(fit_data["logits"][eval_mask])
                ).numpy()

    raw_probs = 1.0 / (1.0 + np.exp(-eval_data["logits"]))
    scaled_probs = 1.0 / (1.0 + np.exp(-scaled_logits))
    payload = {
        "protocol": args.protocol,
        "fit_predictions": str(Path(args.fit_predictions).resolve()),
        "eval_predictions": str(
            Path(args.eval_predictions or args.fit_predictions).resolve()
        ),
        "seed": args.seed,
        "folds": args.folds if args.protocol == "crossfit" else None,
        "temperatures": temperatures,
        "raw": {
            "metrics": compute_metrics(raw_probs, eval_data["targets"]),
            "calibration": compute_calibration_metrics(
                raw_probs, eval_data["targets"], args.num_bins
            ),
        },
        "temperature_scaled": {
            "metrics": compute_metrics(scaled_probs, eval_data["targets"]),
            "calibration": compute_calibration_metrics(
                scaled_probs, eval_data["targets"], args.num_bins
            ),
        },
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "temperature_scaling.json", payload)
    np.savez_compressed(
        output_dir / "temperature_scaled_predictions.npz",
        logits=scaled_logits,
        probabilities=scaled_probs,
        targets=eval_data["targets"],
        indices=eval_data["indices"],
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
