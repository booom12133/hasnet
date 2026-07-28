"""Evaluate probability-level average/max fusion of OL-only and SD-only runs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from hasnet.checkpoint import write_json
from hasnet.metrics import compute_calibration_metrics, compute_metrics


def load(path: str):
    with np.load(path, allow_pickle=True) as archive:
        return (
            archive["probabilities"],
            archive["targets"],
            archive["indices"],
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ol", required=True)
    parser.add_argument("--sd", required=True)
    parser.add_argument("--mode", choices=["avg", "max"], required=True)
    parser.add_argument("--output-dir", default="outputs/late_fusion")
    args = parser.parse_args()
    ol_probs, ol_targets, ol_indices = load(args.ol)
    sd_probs, sd_targets, sd_indices = load(args.sd)
    if not np.array_equal(ol_indices, sd_indices) or not np.array_equal(
        ol_targets, sd_targets
    ):
        raise ValueError("OL and SD archives are not aligned")
    probabilities = (
        (ol_probs + sd_probs) / 2.0
        if args.mode == "avg"
        else np.maximum(ol_probs, sd_probs)
    )
    payload = {
        "mode": args.mode,
        "ol": str(Path(args.ol).resolve()),
        "sd": str(Path(args.sd).resolve()),
        "metrics": compute_metrics(probabilities, ol_targets),
        "calibration": compute_calibration_metrics(probabilities, ol_targets, 10),
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_dir / f"late_fusion_{args.mode}.npz",
        probabilities=probabilities,
        targets=ol_targets,
        indices=ol_indices,
    )
    write_json(output_dir / f"late_fusion_{args.mode}.json", payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
