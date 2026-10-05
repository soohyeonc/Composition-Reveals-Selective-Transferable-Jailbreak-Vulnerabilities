#!/usr/bin/env python3
"""Maintainer-only extraction of paper evidence from a preserved research archive.

No inference. End users use the exported support data, not the private source map.
The immutable release supplies all target/pair/run allowlists and response hashes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

from release_inputs import DEFAULT_RELEASE, MODELS, check_manifest

DISCOVERY = dict(zip(MODELS, [
    "gpt-5.6-luna_full_20260807", "deepseek-r1-8b_full_20260811",
    "gemma4-31b_full_20260814", "qwen3.8-27b_full_20260827",
]))


def digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def csv(path, frame):
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, compression={"method": "gzip", "mtime": 0}
                 if path.suffix == ".gz" else None)


def sql(path, table):
    with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as connection:
        return pd.read_sql_query(f"SELECT * FROM {table}", connection)


def truth(series):
    values = series.astype(str).str.lower().str.strip()
    if not values.isin(["true", "false", "1", "0", "1.0", "0.0"]).all():
        raise ValueError("Invalid persistence label")
    return values.isin(["true", "1", "1.0"])


def build(archive, source_map, release, output, diagram):
    check_manifest(release)
    if output.exists():
        raise ValueError("Refusing to overwrite an existing support dataset")
    output.mkdir(parents=True)
    settings = json.loads((release / "run_settings.json").read_text())
    hashes = json.loads((release / "provenance/source_hashes.json").read_text())
    sources = json.loads(source_map.read_text())
    provenance = []
    for setting in settings:
        model, phase = setting["model"], setting["phase"]
        slug = next(k for k, v in MODELS.items() if v == model)
        if phase == "discovery":
            db = archive / "results/experiments" / DISCOVERY[slug] / "state.sqlite3"
            prefix = slug
        else:
            spec = sources[model]
            db = archive / spec["runs"] / spec["names"][setting["run"] - 1] / "state.sqlite3"
            prefix = f"{slug}_run{setting['run']}"
        actual = digest(db)
        if actual != hashes[setting["source_artifact_id"]]:
            raise ValueError(f"Changed canonical source: {prefix}")
        provenance.append({"phase": phase, "run": prefix, "sha256": actual,
                           "source_artifact_id": setting["source_artifact_id"]})
        for table in ("baselines", "chains"):
            keys = ["prompt_row"] + (["mutator"] if table == "baselines" else ["mutator_1", "mutator_2"])
            reference = pd.read_csv(release / phase / f"{prefix}_{table}.csv.gz", keep_default_na=False)
            frame = sql(db, table)
            # Inner join discards unreported pairs and unused component baselines.
            frame = reference[keys + ["target_response_sha256"]].merge(
                frame[keys + ["target_response"]], on=keys, validate="one_to_one")
            assert len(frame) == len(reference)
            frame["target_response"] = frame.target_response.fillna("")
            observed = frame.target_response.map(lambda x: hashlib.sha256(x.encode()).hexdigest())
            assert observed.eq(frame.target_response_sha256).all(), prefix
            frame = frame.drop(columns="target_response_sha256")
            dest = output / "responses" / phase / prefix
            if table == "baselines":
                csv(dest / "baselines.csv.gz", frame)
            else:
                for mutator, group in frame.groupby("mutator_1", sort=True):
                    csv(dest / f"chains_{mutator}.csv.gz", group)
        print(f"Exported hash-matched responses: {phase}/{prefix}", flush=True)

    # Historical target names identify mutation batches only. No historical
    # target outputs or ASR measurements are included in these exports.
    batch_names = {"deepseek-chat": "a", "gpt-3.5-turbo": "b", "gpt-4-turbo": "c"}
    pattern = re.compile(r"^(?P<m1>[^_]+)_(?P<m2>[^_]+)_(?P<batch>.+)_\d{8}_\d{6}\.csv$")
    batches = {k: [] for k in "abc"}
    historical_hashes = []
    for path in sorted((archive / "results/runs/processed/completed").glob("*.csv")):
        match = pattern.match(path.name)
        if not match or match["batch"] not in batch_names:
            continue
        batch = batch_names[match["batch"]]
        frame = pd.read_csv(path, keep_default_na=False).rename(columns={
            "Input Prompt": "input_prompt", "Mutator 1": "mutator_1", "Mutator 2": "mutator_2",
            "Jailbreak Prompt 1": "jailbreak_prompt_1", "Jailbreak Prompt 2": "jailbreak_prompt_2",
            "M1_Persistence": "m1_persistence", "M2_Persistence": "m2_persistence"})
        cols = ["mutator_1", "mutator_2", "input_prompt", "jailbreak_prompt_1",
                "jailbreak_prompt_2", "m1_persistence", "m2_persistence"]
        frame = frame[cols].copy()
        frame.insert(0, "source_row", range(len(frame)))
        batches[batch].append(frame)
        historical_hashes.append({"batch": batch.upper(), "mutator_1": match["m1"],
                                  "mutator_2": match["m2"], "sha256": digest(path)})
    reference = pd.read_csv(release / "completeness/pair_metrics.csv").set_index(["m1", "m2"])
    for batch, frames in batches.items():
        data = pd.concat(frames, ignore_index=True)
        data["complete"] = truth(data.m1_persistence) & truth(data.m2_persistence)
        rates = data.groupby(["mutator_1", "mutator_2"]).complete.mean()
        assert len(rates) == 132 and not data.duplicated(["mutator_1", "mutator_2", "source_row"]).any()
        assert np.allclose(rates, reference.loc[rates.index, f"batch_{batch}_rate"])
        csv(output / "completeness" / f"batch_{batch.upper()}.csv.gz", data.drop(columns="complete"))

    pilot_db = archive / "results/reproduction/mutator_alignment_gpt56_vs_gpt35_20260902_n20/state.sqlite3"
    assert digest(pilot_db) in hashes.values()
    pilot = sql(pilot_db, "chains")
    keys = ["prompt_row", "mutator_1", "mutator_2"]
    labels = pd.read_csv(release / "writer_pilot/row_labels.csv", keep_default_na=False)
    selected = pilot[keys + ["input_prompt", "jailbreak_prompt_1", "jailbreak_prompt_2",
                            "m1_persistence", "m2_persistence", "provider_status", "status"]].copy()
    joined = labels.merge(selected, on=keys, suffixes=("_released", ""), validate="one_to_one")
    assert len(joined) == 2640
    for pos in ["m1", "m2"]:
        assert truth(joined[f"{pos}_persistence"]).eq(truth(joined[f"{pos}_persistence_candidate"])).all()
    csv(output / "writer_pilot/candidate_chains.csv.gz", selected)
    with sqlite3.connect(f"file:{pilot_db}?mode=ro", uri=True) as connection:
        metadata = {key: json.loads(value) for key, value in connection.execute("SELECT key,value FROM metadata")}
    whitelist = ["sample_seed", "sample_prompt_rows", "sample_prompts", "candidate_mutator_model",
                 "reference_mutator_model", "evaluator_model", "mutator_prompts_sha256",
                 "persistence_prompts_sha256", "alignment_rule"]
    (output / "writer_pilot/settings.json").write_text(json.dumps({k: metadata[k] for k in whitelist}, indent=2) + "\n")

    validations = archive / "results/reproduction"
    source = validations / "intent_judge_validation_20260914/source/strongreject_labelbox.csv"
    assert digest(source) in hashes.values()
    intent = pd.read_csv(source)
    # Opaque cluster IDs preserve first-appearance order for the exact bootstrap,
    # without redistributing the external response corpus under uncertain terms.
    metadata = intent[["model", "jailbreak", "label_median"]].copy()
    metadata.insert(0, "cluster_id", pd.factorize(intent.forbidden_prompt, sort=False)[0])
    metadata.insert(0, "row_id", range(len(metadata)))
    csv(output / "validation/intent_source_metadata.csv", metadata)
    source = validations / "persistence_judge_validation_20260914/source/persistence_check_gt.csv"
    assert digest(source) in hashes.values()
    shutil.copyfile(source, output / "validation/persistence_check_gt.csv")
    (output / "figures").mkdir()
    shutil.copyfile(diagram, output / "figures/chained-mutator.png")
    provenance.append({"kind": "writer_pilot", "sha256": digest(pilot_db)})
    (output / "provenance.json").write_text(json.dumps({
        "release_manifest_sha256": digest(release / "manifest.json"),
        "sources": provenance, "historical_mutation_sources": historical_hashes,
        "scope": "Only four targets and seven reported stability conditions. No new inference.",
        "missing_evidence": ["Independent safety-judge benchmark validation is outside the distributed artifact"],
    }, indent=2) + "\n")
    files = {str(p.relative_to(output)): {"sha256": digest(p), "bytes": p.stat().st_size}
             for p in sorted(output.rglob("*")) if p.is_file()}
    (output / "manifest.json").write_text(json.dumps({"files": files}, indent=2) + "\n")
    print(f"Exported {len(files)} supporting files within the documented artifact scope.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--source-map", type=Path, required=True)
    parser.add_argument("--diagram", type=Path, required=True)
    parser.add_argument("--release", type=Path, default=DEFAULT_RELEASE)
    parser.add_argument("--output", type=Path, default=DEFAULT_RELEASE.parent / "paper-support")
    args = parser.parse_args()
    build(args.archive, args.source_map, args.release, args.output, args.diagram)


if __name__ == "__main__":
    main()
