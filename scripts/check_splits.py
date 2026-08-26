"""Validate released split format, counts, hashes, paths, and disjointness."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path


EXPECTED = {
    "dvxray": {
        "num_classes": 15,
        "splits": {
            "train": (11200, "9dd2beb6232ec150f7f5477f7ae9310afd933eb487c9f3cca7084dd8bcb50bf9"),
            "val": (3200, "5bd1b583b2cebcdc7a5c3c6201fbde511475ecbc6f844dd2f1badec6d2a39fd8"),
            "test": (1600, "5f7226fa18138bb0407092379e6479ea8a04b86ebc4b54975b93aa8c369284dc"),
        },
    },
    "ldxray": {
        "num_classes": 12,
        "splits": {
            "train": (99133, "c1a6ce15db03d5ee07755d486ba06dafcd23dea688a7459e0e57dbe7c544a305"),
            "val": (11015, "45650f6d66fe4f17691cd60893a84f80658b17ffdc5f95d92a1c98ebedf984db"),
            "test": (36849, "c6d3b21971c2e90609bce0cdfaebacb28ed16d9d4a7fdd54a0afac2fa5eae500"),
        },
    },
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_dataset(dataset: str, specification: dict) -> dict:
    identifiers: dict[str, set[tuple[str, str]]] = {}
    rows = {}
    num_classes = int(specification["num_classes"])
    for split, (expected_count, expected_hash) in specification["splits"].items():
        path = Path("splits") / dataset / f"{split}.txt"
        lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line]
        pairs = set()
        positives = [0] * num_classes
        for line_number, line in enumerate(lines, 1):
            if re.search(r"(?i)\b[A-Z]:\\", line) or re.search(
                r"(?:^|#)/(?:home|root)/", line
            ):
                raise ValueError(f"{path}:{line_number}: machine-specific absolute path")
            parts = line.split("#")
            if len(parts) < 3:
                raise ValueError(f"{path}:{line_number}: fewer than three fields")
            labels = [int(value) for value in parts[2].split(",")]
            if len(labels) != num_classes or any(value not in {0, 1} for value in labels):
                raise ValueError(f"{path}:{line_number}: invalid label vector")
            positives = [left + right for left, right in zip(positives, labels)]
            pair = (parts[0], parts[1])
            if pair in pairs:
                raise ValueError(f"{path}:{line_number}: duplicate pair")
            pairs.add(pair)
        digest = sha256(path)
        if len(lines) != expected_count or digest != expected_hash:
            raise ValueError(
                f"{dataset}/{split} changed: count={len(lines)}, sha256={digest}"
            )
        identifiers[split] = pairs
        rows[split] = {
            "count": len(lines),
            "sha256": digest,
            "positive_counts": positives,
        }
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = identifiers[left] & identifiers[right]
        if overlap:
            raise ValueError(f"{dataset} {left}/{right} overlap: {len(overlap)} pairs")
    return rows


def main() -> None:
    report = {
        dataset: validate_dataset(dataset, specification)
        for dataset, specification in EXPECTED.items()
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
