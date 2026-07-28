"""Print or execute the canonical three-seed experiment matrix."""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

import yaml


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group", required=True)
    parser.add_argument("--manifest", default="configs/paper/experiments.yaml")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--data-root", default=".")
    parser.add_argument("--nproc-per-node", type=int, default=2)
    parser.add_argument("--seeds", type=int, nargs="*", default=None)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    manifest = yaml.safe_load(Path(args.manifest).read_text(encoding="utf-8"))
    if args.group not in manifest["groups"]:
        raise ValueError(f"Unknown group {args.group!r}; choose from {sorted(manifest['groups'])}")
    seeds = args.seeds or manifest["canonical_seeds"]
    commands = []
    for config in manifest["groups"][args.group]:
        for seed in seeds:
            if args.nproc_per_node > 1:
                command = [
                    sys.executable,
                    "-m",
                    "torch.distributed.run",
                    "--standalone",
                    f"--nproc_per_node={args.nproc_per_node}",
                    "scripts/train.py",
                ]
            else:
                command = [sys.executable, "scripts/train.py"]
            command += [
                "--config",
                config,
                "--seed",
                str(seed),
                "--data-root",
                args.data_root,
                "--output-dir",
                args.output_dir,
            ]
            commands.append(command)
    for command in commands:
        print(" ".join(shlex.quote(part) for part in command), flush=True)
        if args.execute:
            subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
