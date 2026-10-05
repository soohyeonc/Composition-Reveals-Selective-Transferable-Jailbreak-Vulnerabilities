#!/usr/bin/env python3
"""Run a resumable GPT-5.6 versus GPT-3.5 mutation-alignment pilot.

The pilot regenerates every ordered mutator pair with GPT-5.6 Luna for a
fixed sample of prompt identifiers. It evaluates only mutation persistence.
Target-model generation and safety or intent evaluation are intentionally
excluded.
"""

from __future__ import annotations

import argparse
import json
import random
import sqlite3
from configparser import ConfigParser
from pathlib import Path

import pandas as pd
from openai import BadRequestError

import experiment as core
from release_inputs import DEFAULT_RELEASE, check_manifest
from paper_support import output_directory


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_RUN = DEFAULT_RELEASE
DEFAULT_RUN_DIR = ROOT / "outputs/writer-pilot"
MUTATOR_MODEL = "gpt-5.6-luna"
REFERENCE_MUTATOR_MODEL = "gpt-3.5-turbo-0125"
EVALUATOR_MODEL = "gpt-4o-mini-2024-07-18"
SAMPLE_SEED = 20260902


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--release", dest="source_run", type=Path, default=DEFAULT_SOURCE_RUN)
    parser.add_argument("--sample-prompts", type=int, default=20)
    parser.add_argument("--sample-seed", type=int, default=SAMPLE_SEED)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--rpm-limit", type=int, default=100)
    parser.add_argument("--budget-usd", type=float, default=20.0)
    parser.add_argument("--max-tasks", type=int)
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--init-only", action="store_true")
    parser.add_argument("--execute", action="store_true", help="Explicitly allow paid mutation and judge calls")
    parser.add_argument("--export-only", action="store_true")
    parser.add_argument(
        "--credentials", type=Path, default=core.DEFAULT_CREDENTIALS
    )
    return parser.parse_args()


def read_shared_api_key(path: Path) -> str:
    return core.read_api_key(path)


def load_source_scope(
    source_run: Path, sample_prompts: int, sample_seed: int
) -> tuple[pd.DataFrame, list[int]]:
    check_manifest(source_run)
    keys = ["prompt_row", "mutator_1", "mutator_2"]
    prompts = pd.read_csv(source_run / "prompts/frozen_chains.csv.gz", keep_default_na=False)
    labels = pd.read_csv(source_run / "discovery/luna_chains.csv.gz", keep_default_na=False)
    frame = prompts[keys + ["input_prompt"]].merge(labels, on=keys, validate="one_to_one")
    frame["prompt_key"] = frame.prompt_row.astype(str)
    if not frame["status"].eq("done").all():
        raise RuntimeError("Reference mutation run is incomplete")
    prompt_rows = sorted(frame["prompt_row"].unique().astype(int).tolist())
    if sample_prompts <= 0 or sample_prompts > len(prompt_rows):
        raise ValueError(f"sample-prompts must be between 1 and {len(prompt_rows)}")
    selected = sorted(random.Random(sample_seed).sample(prompt_rows, sample_prompts))
    scope = frame.loc[frame["prompt_row"].isin(selected)].copy()
    expected = sample_prompts * len(core.PAPER_MUTATORS) * (len(core.PAPER_MUTATORS) - 1)
    if len(scope) != expected:
        raise RuntimeError(f"Expected {expected} reference rows, found {len(scope)}")
    return scope, selected


def initialize(
    store: core.Store,
    source_run: Path,
    source_scope: pd.DataFrame,
    selected_prompt_rows: list[int],
    sample_seed: int,
) -> None:
    store.set_metadata(
        {
            "schema_version": 1,
            "created_by": "scripts/run_mutator_alignment_pilot.py",
            "objective": "mutation-engine persistence alignment pilot",
            "source_release_manifest_sha256": core.sha256(source_run / "manifest.json"),
            "reference_mutator_model": REFERENCE_MUTATOR_MODEL,
            "candidate_mutator_model": MUTATOR_MODEL,
            "evaluator_model": EVALUATOR_MODEL,
            "mutator_prompts_sha256": core.sha256(core.DEFAULT_MUTATOR_PROMPTS),
            "persistence_prompts_sha256": core.sha256(core.DEFAULT_PERSISTENCE_PROMPTS),
            "sample_seed": sample_seed,
            "sample_prompt_rows": selected_prompt_rows,
            "sample_prompts": len(selected_prompt_rows),
            "mutators": list(core.PAPER_MUTATORS),
            "pairs": [
                [m1, m2]
                for m1 in core.PAPER_MUTATORS
                for m2 in core.PAPER_MUTATORS
                if m1 != m2
            ],
            "alignment_rule": {
                "aligned": {
                    "spearman_min": 0.80,
                    "mean_absolute_difference_max": 0.10,
                    "median_gate_agreement_min": 0.80,
                },
                "materially_different": {
                    "spearman_below": 0.65,
                    "mean_absolute_difference_above": 0.15,
                    "median_gate_agreement_below": 0.70,
                },
            },
            "prices_usd_per_million": {
                MUTATOR_MODEL: core.PRICES[MUTATOR_MODEL],
                EVALUATOR_MODEL: core.PRICES[EVALUATOR_MODEL],
            },
        }
    )
    unique_inputs = source_scope[
        ["prompt_row", "prompt_key", "input_prompt"]
    ].drop_duplicates()
    input_lookup = {
        int(row.prompt_row): (str(row.prompt_key), str(row.input_prompt))
        for row in unique_inputs.itertuples(index=False)
    }
    rows = []
    for prompt_row in selected_prompt_rows:
        prompt_key, input_prompt = input_lookup[prompt_row]
        for m1 in core.PAPER_MUTATORS:
            for m2 in core.PAPER_MUTATORS:
                if m1 == m2:
                    continue
                rows.append(
                    (
                        core.stable_id("mutator-alignment", sample_seed, prompt_row, m1, m2),
                        prompt_row,
                        prompt_key,
                        input_prompt,
                        m1,
                        m2,
                    )
                )
    store.insert_chains(rows)


class AlignmentExperiment(core.Experiment):
    def mark_mutator_block(self, task_id: str, stage: str) -> None:
        marker = f"[BLOCKED_BY_PROVIDER:{stage}:bio_policy]"
        with self.store.lock:
            self.store.conn.execute(
                """
                UPDATE chains
                SET jailbreak_prompt_1 = COALESCE(jailbreak_prompt_1, ?),
                    jailbreak_prompt_2 = ?,
                    m1_persistence = 'FALSE', m2_persistence = 'FALSE',
                    provider_status = 'blocked_mutator',
                    provider_error_code = 'bio_policy', status = 'done',
                    last_error = NULL
                WHERE task_id = ?
                """,
                (marker, marker, task_id),
            )
            self.store.conn.commit()

    def process_chain(self, task_id: str) -> None:
        try:
            row = self.store.row("chains", task_id)
            jp1 = row["jailbreak_prompt_1"]
            if jp1 is None:
                try:
                    jp1 = self.call_and_save(
                        table="chains",
                        task_kind="alignment",
                        task_id=task_id,
                        stage="mutator_1",
                        field="jailbreak_prompt_1",
                        model=MUTATOR_MODEL,
                        system_prompt=self.mutator_prompts[row["mutator_1"]],
                        user_prompt=row["input_prompt"],
                        max_tokens=4096,
                        reasoning_effort="none",
                    )
                except BadRequestError as exc:
                    if "bio_policy" in str(exc):
                        self.mark_mutator_block(task_id, "mutator_1")
                        return
                    raise

            row = self.store.row("chains", task_id)
            jp2 = row["jailbreak_prompt_2"]
            if jp2 is None:
                try:
                    jp2 = self.call_and_save(
                        table="chains",
                        task_kind="alignment",
                        task_id=task_id,
                        stage="mutator_2",
                        field="jailbreak_prompt_2",
                        model=MUTATOR_MODEL,
                        system_prompt=self.mutator_prompts[row["mutator_2"]],
                        user_prompt=jp1,
                        max_tokens=4096,
                        reasoning_effort="none",
                    )
                except BadRequestError as exc:
                    if "bio_policy" in str(exc):
                        self.mark_mutator_block(task_id, "mutator_2")
                        return
                    raise

            row = self.store.row("chains", task_id)
            if row["m1_persistence"] is None:
                key = f"{row['mutator_1'][3:]}-persistence-eval"
                self.call_and_save(
                    table="chains",
                    task_kind="alignment",
                    task_id=task_id,
                    stage="m1_persistence",
                    field="m1_persistence",
                    model=EVALUATOR_MODEL,
                    system_prompt=core.PERSISTENCE_SYSTEM_PROMPT,
                    user_prompt=self.persistence_prompts[key].format(
                        original_prompt=row["input_prompt"], final_prompt=jp2
                    ),
                    max_tokens=16,
                    temperature=0,
                )

            row = self.store.row("chains", task_id)
            if row["m2_persistence"] is None:
                key = f"{row['mutator_2'][3:]}-persistence-eval"
                self.call_and_save(
                    table="chains",
                    task_kind="alignment",
                    task_id=task_id,
                    stage="m2_persistence",
                    field="m2_persistence",
                    model=EVALUATOR_MODEL,
                    system_prompt=core.PERSISTENCE_SYSTEM_PROMPT,
                    user_prompt=self.persistence_prompts[key].format(
                        original_prompt=jp1, final_prompt=jp2
                    ),
                    max_tokens=16,
                    temperature=0,
                )
            self.store.mark_done("chains", task_id)
        except core.BudgetExceeded:
            self.stop.set()
            raise
        except Exception as exc:
            self.store.mark_error("chains", task_id, exc)
            raise


def main() -> int:
    args = parse_args()
    if args.workers <= 0 or args.rpm_limit <= 0 or args.budget_usd <= 0:
        raise ValueError("workers, rpm-limit, and budget-usd must be positive")
    source_run = args.source_run.resolve()
    source_scope, selected_prompt_rows = load_source_scope(
        source_run, args.sample_prompts, args.sample_seed
    )
    args.run_dir = output_directory(args.run_dir)
    store = core.Store(args.run_dir / "state.sqlite3")
    initialize(store, source_run, source_scope, selected_prompt_rows, args.sample_seed)
    print(
        json.dumps(
            {
                "run_dir": str(args.run_dir.resolve()),
                "sample_prompt_rows": selected_prompt_rows,
                "pairs": 132,
                "chain_tasks": len(selected_prompt_rows) * 132,
                "models": {
                    "reference_mutator": REFERENCE_MUTATOR_MODEL,
                    "candidate_mutator": MUTATOR_MODEL,
                    "persistence_evaluator": EVALUATOR_MODEL,
                },
                "counts": store.counts()["chains"],
                "recorded_cost_usd": round(store.total_cost(), 6),
                "budget_usd": args.budget_usd,
            },
            indent=2,
        )
    )
    if args.init_only or (not args.execute and not args.export_only):
        print("Initialized only. Add --execute to call the writer and persistence judge.")
        return 0
    if args.export_only:
        store.export(args.run_dir / "exports")
        return 0

    api_key = read_shared_api_key(args.credentials)
    mutator_prompts = core.load_yaml(core.DEFAULT_MUTATOR_PROMPTS)
    persistence_prompts = core.load_yaml(core.DEFAULT_PERSISTENCE_PROMPTS)
    experiment = AlignmentExperiment(
        store,
        api_key,
        mutator_prompts,
        {},
        persistence_prompts,
        core.Budget(store.total_cost(), args.budget_usd),
        core.RateLimiter(args.rpm_limit),
    )
    core.run_tasks(
        experiment,
        "chains",
        args.workers,
        args.retry_failed,
        args.max_tasks,
    )
    store.export(args.run_dir / "exports")
    return 0 if store.counts()["chains"].get("done", 0) == len(selected_prompt_rows) * 132 else 2


if __name__ == "__main__":
    raise SystemExit(main())
