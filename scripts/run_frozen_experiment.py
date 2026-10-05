#!/usr/bin/env python3
"""Initialize or rerun the paper's released frozen inputs. Network calls require --execute."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import requests
import pandas as pd

import experiment as core
from ollama_backend import OllamaExperiment
from release_inputs import (DEFAULT_RELEASE, MODELS, check_manifest,
                            initialize_store, load_inputs)
from paper_support import SUPPORT, check_support, output_directory


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", type=Path, default=DEFAULT_RELEASE)
    parser.add_argument("--target", choices=MODELS, required=True)
    parser.add_argument("--scope", choices=["discovery", "stability"], default="stability")
    parser.add_argument("--run", type=int, choices=range(1, 6), default=1)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-prompts", type=int, default=520,
                        help="Smoke-test subset; fewer than 520 is not paper reproduction")
    parser.add_argument("--execute", action="store_true", help="Permit paid judge/target requests")
    parser.add_argument("--rejudge-archived", action="store_true",
                        help="Reuse recorded target responses and call only safety/intent judges")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--rpm-limit", type=int, default=300)
    parser.add_argument("--budget-usd", type=float, default=25,
                        help="Estimated stop threshold at historical rates, not a billing cap")
    parser.add_argument("--ollama-url", action="append", default=None)
    parser.add_argument("--credentials", type=Path, default=core.DEFAULT_CREDENTIALS,
                        help="Optional INI fallback; prefer OPENAI_API_KEY")
    parser.add_argument("--target-timeout", type=float, default=900)
    parser.add_argument("--retry-failed", action="store_true")
    return parser.parse_args()


def check_ollama(urls, settings):
    expected = settings.get("target_model_digest")
    model = settings["target_model"]
    if not expected:
        raise ValueError("Recorded Ollama checkpoint digest is missing")
    for url in urls:
        response = requests.get(url.rstrip("/") + "/api/tags", timeout=20)
        response.raise_for_status()
        found = [m for m in response.json()["models"]
                 if m.get("name") == model or m.get("model") == model]
        if len(found) != 1 or not found[0].get("digest", "").startswith(expected):
            raise ValueError(f"{model} at {url} does not match the recorded checkpoint")


def attach_archived_responses(store, release, target, scope, run, base, chains):
    prefix = target if scope == "discovery" else f"{target}_run{run}"
    folder = SUPPORT / "responses" / scope / prefix
    for table, tasks in [("baselines", base), ("chains", chains)]:
        keys = ["prompt_row"] + (["mutator"] if table == "baselines" else ["mutator_1", "mutator_2"])
        responses = pd.concat([pd.read_csv(p, keep_default_na=False)
                               for p in sorted(folder.glob(f"{table}*.csv.gz"))], ignore_index=True)
        labels = pd.read_csv(release / scope / f"{prefix}_{table}.csv.gz", keep_default_na=False)
        rows = tasks[keys + ["task_id"]].merge(responses, on=keys, validate="one_to_one").merge(
            labels[keys + ["provider_status", "target_response_sha256"]], on=keys, validate="one_to_one")
        if len(rows) != len(tasks):
            raise ValueError("Archived responses do not cover all requested tasks")
        import hashlib
        if not rows.target_response.map(lambda x: hashlib.sha256(x.encode()).hexdigest()).eq(rows.target_response_sha256).all():
            raise ValueError("Archived target response hash mismatch")
        for row in rows.itertuples():
            store.conn.execute(f"UPDATE {table} SET target_response=? WHERE task_id=? AND target_response IS NULL",
                               (row.target_response, row.task_id))
            if str(row.provider_status).startswith("blocked"):
                store.mark_provider_block(table, row.task_id, "bio_policy")
        store.conn.commit()


def main():
    args = parse_args()
    if min(args.workers, args.rpm_limit, args.budget_usd, args.target_timeout) <= 0:
        raise ValueError("Worker, rate, budget and timeout values must be positive")
    release = args.release.resolve()
    output = args.output_dir.resolve()
    if output.is_relative_to(release) or release.is_relative_to(output):
        raise ValueError("Output must be separate from the immutable release")
    output = output_directory(output)
    manifest_hash = check_manifest(release)
    base, chains, record = load_inputs(release, args.target, args.scope, args.run, args.max_prompts)
    settings = record["settings"]
    core.TARGET_MODEL = settings["target_model"]
    core.EVALUATOR_MODEL = settings["evaluator_model"]
    provider = "openai" if args.target == "luna" else "ollama"
    if provider == "ollama":
        core.PRICES[core.TARGET_MODEL] = {"input": 0, "cached": 0, "output": 0}
    # 8192 is the archived runner default where the discovery metadata omit it.
    effective_context = settings.get("target_num_ctx", 8192)
    urls = args.ollama_url or ["http://127.0.0.1:11434"]
    metadata = {
        "created_by": "scripts/run_frozen_experiment.py",
        "release_manifest_sha256": manifest_hash,
        "target": args.target, "scope": args.scope,
        "run": args.run if args.scope == "stability" else None,
        "max_prompts": args.max_prompts,
        "recorded_settings": settings,
        "effective_target_num_ctx": effective_context if provider == "ollama" else None,
        "context_source": ("recorded" if "target_num_ctx" in settings else "archived_runner_default")
                          if provider == "ollama" else None,
        "frozen_mutations": True, "frozen_persistence": True,
        "rejudge_archived_responses": args.rejudge_archived,
        "support_manifest_sha256": check_support() if args.rejudge_archived else None,
        "accounting_rates_usd_per_million": core.PRICES,
    }
    output.mkdir(parents=True, exist_ok=True)
    store = core.Store(output / "state.sqlite3")
    try:
        initialize_store(store, base, chains, metadata)
        if args.rejudge_archived:
            attach_archived_responses(store, release, args.target, args.scope, args.run, base, chains)
        plan = {"target": core.TARGET_MODEL, "scope": args.scope,
                "baseline_tasks": len(base), "chain_tasks": len(chains),
                "pairs": chains.groupby(["mutator_1", "mutator_2"]).ngroups,
                "prompts_per_condition": args.max_prompts,
                "execute": args.execute, "counts": store.counts()}
        print(json.dumps(plan, indent=2))
        if not args.execute:
            print("Initialized only. No credentials read and no network requests made.")
            return 0
        api_key = core.read_api_key(args.credentials)
        if provider == "ollama" and not args.rejudge_archived:
            check_ollama(urls, settings)
        common = (store, api_key,
                  core.load_yaml(release / "prompts/mutator-system-prompts.yaml"),
                  core.load_yaml(release / "prompts/evaluator-system-prompts.yaml"),
                  core.load_yaml(release / "prompts/persistence-eval-prompts.yaml"),
                  core.Budget(store.total_cost(), args.budget_usd), core.RateLimiter(args.rpm_limit))
        if provider == "ollama" and not args.rejudge_archived:
            experiment = OllamaExperiment(
                *common, ollama_urls=urls, target_model=core.TARGET_MODEL,
                target_timeout=args.target_timeout, think=settings.get("target_think", False),
                num_ctx=effective_context, target_seed=settings.get("target_seed"))
        else:
            experiment = core.Experiment(*common)
        for table in ["baselines", "chains"]:
            if not experiment.stop.is_set():
                core.run_tasks(experiment, table, args.workers, args.retry_failed, None)
        store.export(output / "exports")
        counts = store.counts()
        complete = all(sum(v for k, v in group.items() if k != "done") == 0
                       for group in counts.values())
        summary = {"counts": counts, "complete": complete,
                   "historical_rate_cost_estimate_usd": store.total_cost(),
                   "stopped_for_budget": experiment.stop.is_set()}
        (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps(summary, indent=2))
        return 0 if complete else 2
    finally:
        store.conn.close()


if __name__ == "__main__":
    raise SystemExit(main())
