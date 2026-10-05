"""Shared safeguards for extended, offline paper reproduction workflows."""
from __future__ import annotations

import json
from pathlib import Path

from release_inputs import ROOT, sha256

SUPPORT = ROOT / "release/paper-support"


def check_support(path=SUPPORT):
    path = Path(path).resolve()
    manifest = json.loads((path / "manifest.json").read_text())
    actual = {str(p.relative_to(path)) for p in path.rglob("*")
              if p.is_file() and p.name != "manifest.json"}
    if actual != set(manifest["files"]):
        raise ValueError("Support dataset inventory mismatch")
    for name, spec in manifest["files"].items():
        file = path / name
        if file.is_symlink() or not file.resolve().is_relative_to(path):
            raise ValueError(f"Invalid support path: {name}")
        if sha256(file) != spec["sha256"] or file.stat().st_size != spec["bytes"]:
            raise ValueError(f"Support dataset hash mismatch: {name}")
    return sha256(path / "manifest.json")


def output_directory(path):
    path = Path(path).resolve()
    protected = [ROOT / "release", ROOT / "results", ROOT / "scripts", ROOT / "tests"]
    if path == ROOT or any(path.is_relative_to(p) or p.is_relative_to(path) for p in protected):
        raise ValueError("Use an output directory outside the published data, results, and code")
    path.mkdir(parents=True, exist_ok=True)
    return path


def lock_run(path, settings):
    path = output_directory(path)
    record = path / "run_identity.json"
    if record.exists() and json.loads(record.read_text()) != settings:
        raise ValueError("Run inputs or settings changed; use a new output directory")
    if not record.exists() and list(path.glob("*.sqlite3")):
        raise ValueError("Refusing to resume an unidentified checkpoint")
    record.write_text(json.dumps(settings, indent=2, sort_keys=True) + "\n")
