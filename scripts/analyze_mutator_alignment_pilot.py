#!/usr/bin/env python3
"""Analyze the GPT-5.6 versus GPT-3.5 mutation-alignment pilot."""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
from experiment import PAPER_MUTATORS
from release_inputs import DEFAULT_RELEASE
from paper_support import SUPPORT, output_directory


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_DIR = ROOT / "outputs/writer-pilot"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, help="Analyze a fresh completed writer run instead of released evidence")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_RUN_DIR / "analysis")
    parser.add_argument("--release", type=Path, default=DEFAULT_RELEASE)
    parser.add_argument("--support", type=Path, default=SUPPORT)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260902)
    return parser.parse_args()


def parse_label(value: object) -> bool:
    label = str(value).strip().upper()
    if label == "TRUE":
        return True
    if label == "FALSE":
        return False
    raise ValueError(f"Unexpected persistence label: {value!r}")


def correlation(left: np.ndarray, right: np.ndarray, method: str) -> float:
    return float(pd.Series(left).corr(pd.Series(right), method=method))


def summarize_rates(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    reference_gate = reference >= np.median(reference)
    candidate_gate = candidate >= np.median(candidate)
    union = np.logical_or(reference_gate, candidate_gate).sum()
    return {
        "pearson": correlation(reference, candidate, "pearson"),
        "spearman": correlation(reference, candidate, "spearman"),
        "mean_absolute_difference": float(np.abs(reference - candidate).mean()),
        "mean_signed_difference": float((candidate - reference).mean()),
        "median_gate_agreement": float((reference_gate == candidate_gate).mean()),
        "median_gate_jaccard": float(
            np.logical_and(reference_gate, candidate_gate).sum() / union
        ),
    }


def decision(metrics: dict[str, float], rule: dict[str, dict[str, float]]) -> str:
    aligned = rule["aligned"]
    different = rule["materially_different"]
    if (
        metrics["spearman"] >= aligned["spearman_min"]
        and metrics["mean_absolute_difference"]
        <= aligned["mean_absolute_difference_max"]
        and metrics["median_gate_agreement"]
        >= aligned["median_gate_agreement_min"]
    ):
        return "aligned"
    if (
        metrics["spearman"] < different["spearman_below"]
        or metrics["mean_absolute_difference"]
        > different["mean_absolute_difference_above"]
        or metrics["median_gate_agreement"]
        < different["median_gate_agreement_below"]
    ):
        return "materially_different"
    return "inconclusive_expand_pilot"


def main() -> int:
    args = parse_args()
    if args.run_dir is not None:
        database = args.run_dir / "state.sqlite3"
        with sqlite3.connect(f"file:{database}?mode=ro", uri=True) as connection:
            pilot = pd.read_sql_query("SELECT * FROM chains", connection)
            metadata = {key: json.loads(value) for key, value in connection.execute("SELECT key, value FROM metadata")}
    else:
        pilot = pd.read_csv(args.support / "writer_pilot/candidate_chains.csv.gz", keep_default_na=False)
        metadata = json.loads((args.support / "writer_pilot/settings.json").read_text())
        metadata["mutators"] = list(PAPER_MUTATORS)
    if not pilot["status"].eq("done").all():
        counts = pilot["status"].value_counts().to_dict()
        raise RuntimeError(f"Pilot is incomplete: {counts}")

    reference = pd.read_csv(args.release / "discovery/luna_chains.csv.gz", usecols=[
        "prompt_row", "mutator_1", "mutator_2", "m1_persistence", "m2_persistence"])
    key_columns = ["prompt_row", "mutator_1", "mutator_2"]
    reference = reference.loc[
        reference["prompt_row"].isin(metadata["sample_prompt_rows"])
    ].copy()
    merged = pilot.merge(
        reference,
        on=key_columns,
        how="left",
        validate="one_to_one",
        suffixes=("_candidate", "_reference"),
    )
    if merged[["m1_persistence_reference", "m2_persistence_reference"]].isna().any().any():
        raise RuntimeError("Reference persistence labels are missing")

    for engine in ("reference", "candidate"):
        for position in ("m1", "m2"):
            column = f"{position}_persistence_{engine}"
            merged[f"{position}_{engine}"] = merged[column].map(parse_label)
        merged[f"complete_{engine}"] = (
            merged[f"m1_{engine}"] & merged[f"m2_{engine}"]
        )

    pair_columns = ["mutator_1", "mutator_2"]
    pair_metrics = (
        merged.groupby(pair_columns, sort=True)
        .agg(
            prompts=("prompt_row", "size"),
            reference_complete=("complete_reference", "sum"),
            candidate_complete=("complete_candidate", "sum"),
        )
        .reset_index()
    )
    pair_metrics["reference_rate"] = (
        pair_metrics["reference_complete"] / pair_metrics["prompts"]
    )
    pair_metrics["candidate_rate"] = (
        pair_metrics["candidate_complete"] / pair_metrics["prompts"]
    )
    pair_metrics["rate_difference"] = (
        pair_metrics["candidate_rate"] - pair_metrics["reference_rate"]
    )

    reference_rates = pair_metrics["reference_rate"].to_numpy(float)
    candidate_rates = pair_metrics["candidate_rate"].to_numpy(float)
    point = summarize_rates(reference_rates, candidate_rates)

    pair_keys = pd.MultiIndex.from_frame(pair_metrics[pair_columns])
    prompt_rows = sorted(merged["prompt_row"].unique().astype(int).tolist())
    reference_matrix = (
        merged.pivot(index="prompt_row", columns=pair_columns, values="complete_reference")
        .reindex(index=prompt_rows, columns=pair_keys)
        .to_numpy(float)
    )
    candidate_matrix = (
        merged.pivot(index="prompt_row", columns=pair_columns, values="complete_candidate")
        .reindex(index=prompt_rows, columns=pair_keys)
        .to_numpy(float)
    )
    rng = np.random.default_rng(args.bootstrap_seed)
    bootstraps: list[dict[str, float]] = []
    for _ in range(args.bootstrap_samples):
        indices = rng.integers(0, len(prompt_rows), size=len(prompt_rows))
        bootstraps.append(
            summarize_rates(
                reference_matrix[indices].mean(axis=0),
                candidate_matrix[indices].mean(axis=0),
            )
        )
    intervals = {}
    for metric in point:
        values = np.array([item[metric] for item in bootstraps], dtype=float)
        values = values[np.isfinite(values)]
        intervals[metric] = {
            "low": float(np.quantile(values, 0.025)),
            "high": float(np.quantile(values, 0.975)),
        }

    per_mutator_rows = []
    for mutator in metadata["mutators"]:
        for position, mutator_column in (("M1", "mutator_1"), ("M2", "mutator_2")):
            group = merged.loc[merged[mutator_column].eq(mutator)]
            pos = position.lower()
            per_mutator_rows.append(
                {
                    "mutator": mutator,
                    "position": position,
                    "n": len(group),
                    "reference_persistence_rate": float(group[f"{pos}_reference"].mean()),
                    "candidate_persistence_rate": float(group[f"{pos}_candidate"].mean()),
                    "rate_difference": float(
                        group[f"{pos}_candidate"].mean()
                        - group[f"{pos}_reference"].mean()
                    ),
                }
            )
    per_mutator = pd.DataFrame(per_mutator_rows)

    output_dir = output_directory(args.output_dir)
    pair_metrics.to_csv(output_dir / "pair_metrics.csv", index=False)
    per_mutator.to_csv(output_dir / "per_mutator_persistence.csv", index=False)
    rule = metadata["alignment_rule"]
    summary = {
        "analysis": "GPT-5.6 versus GPT-3.5 mutation persistence alignment pilot",
        "sample_prompts": len(prompt_rows),
        "pairs": len(pair_metrics),
        "rows": len(merged),
        "candidate_provider_blocks": int(
            pilot["provider_status"].eq("blocked_mutator").sum()
        ),
        "row_level_complete_agreement": float(
            (merged["complete_reference"] == merged["complete_candidate"]).mean()
        ),
        "point_estimates": point,
        "bootstrap_95_percent_intervals": intervals,
        "alignment_rule": rule,
        "decision": decision(point, rule),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
