"""Fail CI on common release-integrity and secret-leak mistakes."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hasnet.config import load_config


FORBIDDEN_PARTS = {
    ".codex_tmp_hasnetzip",
    ".codex_tmp_gvcnn",
    ".codex_tmp_mvcnn",
    ".codex_tmp_opixray",
    ".codex_tmp_sxmnet",
    "scripe",
    "__pycache__",
}
FORBIDDEN_SUFFIXES = {".pt", ".pth", ".ckpt", ".npz", ".npy"}
SECRET_PATTERNS = [
    re.compile(r"\bghp_[A-Za-z0-9]{20,}\b"),
    re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}\b"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"(?i)\b(?:access[_-]?token|api[_-]?key|password)\s*=\s*['\"][^'\"]{8,}['\"]"),
]


def tracked_files() -> list[Path]:
    output = subprocess.check_output(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard"], text=True
    )
    return [Path(line) for line in output.splitlines() if line]


def main() -> None:
    violations = []
    for path in tracked_files():
        if not path.exists():
            continue
        if any(part in FORBIDDEN_PARTS for part in path.parts):
            violations.append(f"forbidden path: {path}")
        if path.suffix.lower() in FORBIDDEN_SUFFIXES:
            violations.append(f"binary artifact must use Releases/external storage: {path}")
        if path.suffix.lower() in {".py", ".yaml", ".yml", ".json", ".md", ".toml", ".txt", ".cff"}:
            text = path.read_text(encoding="utf-8", errors="ignore")
            for pattern in SECRET_PATTERNS:
                if pattern.search(text):
                    violations.append(f"possible credential: {path}")
            if path.suffix.lower() in {".py", ".yaml", ".yml"} and re.search(
                r"(?i)\b[A-Z]:\\", text
            ):
                violations.append(f"machine-specific Windows path: {path}")
    for path in Path("configs/paper").rglob("*.yaml"):
        if path.name != "robustness.yaml":
            load_config(path)
    if violations:
        raise SystemExit("\n".join(sorted(set(violations))))
    print(f"Release audit passed for {len(tracked_files())} tracked files.")


if __name__ == "__main__":
    main()
