"""Validate split format, counts, disjointness, and stable SHA-256 values."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


EXPECTED = {
    "train": (11200, "ac1078e2eedc0e857dafdb45d49c16c76c869c0fcdd1349603567256a477a9d8"),
    "val": (3200, "f32c3e7ab809e73913219f443e5704349cbadf2d31852357cc8935eac25d0459"),
    "test": (1600, "bb0980e60897cd1a776fe149344e018489b7e376b8ab431c239d2b08a8913991"),
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    identifiers: dict[str, set[tuple[str, str]]] = {}
    rows = {}
    for split, (expected_count, expected_hash) in EXPECTED.items():
        path = Path("splits/dvxray") / f"{split}.txt"
        lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line]
        pairs = set()
        positives = [0] * 15
        for line_number, line in enumerate(lines, 1):
            parts = line.split("#")
            if len(parts) < 3:
                raise ValueError(f"{path}:{line_number}: fewer than three fields")
            labels = [int(value) for value in parts[2].split(",")]
            if len(labels) != 15 or any(value not in {0, 1} for value in labels):
                raise ValueError(f"{path}:{line_number}: invalid label vector")
            positives = [left + right for left, right in zip(positives, labels)]
            pair = (parts[0], parts[1])
            if pair in pairs:
                raise ValueError(f"{path}:{line_number}: duplicate pair")
            pairs.add(pair)
        digest = sha256(path)
        if len(lines) != expected_count or digest != expected_hash:
            raise ValueError(
                f"{split} split changed: count={len(lines)}, sha256={digest}"
            )
        identifiers[split] = pairs
        rows[split] = {"count": len(lines), "sha256": digest, "positive_counts": positives}
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = identifiers[left] & identifiers[right]
        if overlap:
            raise ValueError(f"{left}/{right} overlap: {len(overlap)} pairs")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
