"""Plot class-wise macro reliability curves with the paper's 10-bin protocol."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np

from hasnet.metrics import reliability_curve


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--labels", nargs="+", required=True)
    parser.add_argument("--num-bins", type=int, default=10)
    parser.add_argument("--output", default="outputs/calibration/reliability.pdf")
    args = parser.parse_args()
    if len(args.inputs) != len(args.labels):
        raise ValueError("--inputs and --labels must have equal length")

    curves = {}
    fig, axis = plt.subplots(figsize=(5.4, 5.0))
    axis.plot([0, 1], [0, 1], linestyle="--", color="black", linewidth=1, label="Ideal")
    for path, label in zip(args.inputs, args.labels):
        with np.load(path, allow_pickle=True) as archive:
            probabilities = archive["probabilities"]
            targets = archive["targets"]
        curve = reliability_curve(probabilities, targets, args.num_bins)
        curves[label] = curve
        x = np.asarray(curve["confidence"])
        y = np.asarray(curve["empirical_rate"])
        valid = np.isfinite(x) & np.isfinite(y)
        axis.plot(x[valid], y[valid], marker="o", linewidth=1.5, label=label)
    axis.set(xlim=(0, 1), ylim=(0, 1), xlabel="Mean predicted probability", ylabel="Empirical positive rate")
    axis.grid(alpha=0.2)
    axis.legend(frameon=False)
    fig.tight_layout()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    output.with_suffix(".json").write_text(
        json.dumps({"num_bins": args.num_bins, "curves": curves}, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
