"""Read and validate frozen paper inputs without the private research archive."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RELEASE = ROOT / "release/tifs-20261005"
MODELS = {
    "luna": "GPT-5.6 Luna", "r1": "DeepSeek R1 8B",
    "gemma": "Gemma 4 31B", "qwen": "Qwen3.8 27B",
}


def sha256(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def check_manifest(release):
    release = Path(release).resolve()
    manifest_path = release / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    actual = {str(p.relative_to(release)) for p in release.rglob("*")
              if p.is_file() and p != manifest_path}
    if actual != set(manifest["files"]):
        raise ValueError("Release inventory differs from its manifest")
    for relative, digest in manifest["files"].items():
        path = release / relative
        if not path.resolve().is_relative_to(release) or path.is_symlink():
            raise ValueError(f"Invalid release path: {relative}")
        if sha256(path) != digest:
            raise ValueError(f"Release hash mismatch: {relative}")
    return sha256(manifest_path)


def select_settings(release, target, scope, run):
    if target not in MODELS or scope not in {"discovery", "stability"}:
        raise ValueError("Unknown target or scope")
    if not 1 <= run <= 5:
        raise ValueError("Stability run must be between 1 and 5")
    records = json.loads((Path(release) / "run_settings.json").read_text())
    chosen = [r for r in records if r["model"] == MODELS[target]
              and r["phase"] == scope
              and (scope == "discovery" or r["run"] == run)]
    if len(chosen) != 1:
        raise ValueError("Expected exactly one recorded configuration")
    return chosen[0]


def load_inputs(release, target, scope="discovery", run=1, max_prompts=520):
    release = Path(release)
    if not 1 <= max_prompts <= 520:
        raise ValueError("max-prompts must be between 1 and 520")
    record = select_settings(release, target, scope, run)
    base = pd.read_csv(release / "prompts/frozen_baselines.csv.gz", keep_default_na=False)
    chain = pd.read_csv(release / "prompts/frozen_chains.csv.gz", keep_default_na=False)
    keys = ["prompt_row", "mutator_1", "mutator_2"]
    labels = pd.read_csv(release / f"discovery/{target}_chains.csv.gz",
                         usecols=keys + ["m1_persistence", "m2_persistence"],
                         keep_default_na=False)
    if len(base) != 6240 or len(chain) != 68640 or len(labels) != 68640:
        raise ValueError("Incomplete frozen source")
    chain = chain.merge(labels, on=keys, validate="one_to_one")
    if scope == "stability":
        panel = json.loads((release / "panel.json").read_text())["models"][MODELS[target]]
        pairs = {(a, b) for _, a, b in panel}
        if len(pairs) != 7:
            raise ValueError("The reported panel must contain seven conditions")
        chain = chain.loc[[(a, b) in pairs for a, b in zip(chain.mutator_1, chain.mutator_2)]]
        components = {m for pair in pairs for m in pair}
        base = base.loc[base.mutator.isin(components)]
    base = base.loc[base.prompt_row < max_prompts].copy()
    chain = chain.loc[chain.prompt_row < max_prompts].copy()
    for frame, group, texts in [
        (base, ["mutator"], ["input_prompt", "mutated_prompt"]),
        (chain, ["mutator_1", "mutator_2"], ["input_prompt", "jailbreak_prompt_2"]),
    ]:
        if frame.duplicated(group + ["prompt_row"]).any():
            raise ValueError("Duplicate frozen input")
        for _, rows in frame.groupby(group):
            if set(rows.prompt_row) != set(range(max_prompts)):
                raise ValueError("Unequal prompt coverage")
        if any(frame[column].astype(str).str.strip().eq("").any() for column in texts):
            raise ValueError("Empty frozen prompt")
    # One released intermediate is empty. Preserve it exactly; the nonempty
    # final prompt is the target input. Do not drop the row or regenerate it.
    for position in [1, 2]:
        column = f"m{position}_persistence"
        normalized = chain[column].astype(str).str.lower()
        if not normalized.isin(["true", "false", "1", "0"]).all():
            raise ValueError("Invalid frozen persistence label")
        chain[column] = normalized.isin(["true", "1"]).map({True: "TRUE", False: "FALSE"})
    # Keep the original research runner's task identities and seed derivation.
    def task_id(kind, *parts):
        return hashlib.sha256("\x1f".join(map(str, (kind,) + parts)).encode()).hexdigest()
    base["task_id"] = [task_id("baseline", r.prompt_row, r.mutator) for r in base.itertuples()]
    chain["task_id"] = [task_id("chain", r.prompt_row, r.mutator_1, r.mutator_2)
                        for r in chain.itertuples()]
    for frame in [base, chain]:
        frame["prompt_key"] = frame.prompt_row.astype(str)
    return base, chain, record


def initialize_store(store, base, chain, metadata):
    """Idempotent initialization; changed inputs/settings cannot silently resume."""
    store.set_metadata(metadata)
    store.insert_baselines((r.task_id, int(r.prompt_row), r.prompt_key, r.input_prompt, r.mutator)
                           for r in base.itertuples())
    store.insert_chains((r.task_id, int(r.prompt_row), r.prompt_key, r.input_prompt,
                         r.mutator_1, r.mutator_2) for r in chain.itertuples())
    with store.lock:
        store.conn.executemany("UPDATE baselines SET mutated_prompt=? WHERE task_id=?",
                               ((r.mutated_prompt, r.task_id) for r in base.itertuples()))
        store.conn.executemany(
            "UPDATE chains SET jailbreak_prompt_1=?, jailbreak_prompt_2=?, "
            "m1_persistence=?, m2_persistence=? WHERE task_id=?",
            ((r.jailbreak_prompt_1, r.jailbreak_prompt_2, r.m1_persistence,
              r.m2_persistence, r.task_id) for r in chain.itertuples()))
        store.conn.commit()
