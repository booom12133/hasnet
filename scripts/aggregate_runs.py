"""Aggregate per-seed summaries into mean and sample standard deviation."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", default="outputs")
    parser.add_argument("--output", default="results/generated/three_seed_summary.csv")
    args = parser.parse_args()
    grouped: dict[str, list[dict]] = defaultdict(list)
    for path in Path(args.input_dir).rglob("summary.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["_path"] = str(path)
        grouped[payload["experiment"]].append(payload)
    rows = []
    for experiment, runs in sorted(grouped.items()):
        seeds = [int(run["seed"]) for run in runs]
        if len(seeds) != len(set(seeds)):
            raise ValueError(f"Duplicate seeds for {experiment}: {seeds}")
        val = np.asarray([run["best_val_mAP"] for run in runs], dtype=float)
        test = np.asarray([run["test"]["mAP"] for run in runs], dtype=float)
        rows.append(
            {
                "experiment": experiment,
                "n_runs": len(runs),
                "seeds": ";".join(map(str, sorted(seeds))),
                "val_mAP_mean": val.mean(),
                "val_mAP_std": val.std(ddof=1) if len(val) > 1 else 0.0,
                "test_mAP_mean": test.mean(),
                "test_mAP_std": test.std(ddof=1) if len(test) > 1 else 0.0,
            }
        )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["experiment"])
        writer.writeheader()
        writer.writerows(rows)
    output.with_suffix(".json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"Wrote {len(rows)} experiments to {output}")


if __name__ == "__main__":
    main()
