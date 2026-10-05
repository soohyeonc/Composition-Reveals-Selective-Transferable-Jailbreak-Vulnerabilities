#!/usr/bin/env python3
"""Validate persistence judges against the existing human annotations.

The source has 132 ordered-pair rows with one human label for each mutator
position. The runner expands these rows into 264 decisions and checkpoints one
judge at a time in SQLite so provider failures can be resumed safely.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import re
import shutil
import sqlite3
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import pandas as pd
import requests
import yaml
from openai import OpenAI
from release_inputs import DEFAULT_RELEASE, check_manifest
from paper_support import SUPPORT, check_support, lock_run, output_directory
from run_intent_judge_validation import load_key


SOURCE_PATH = SUPPORT / "validation/persistence_check_gt.csv"
DEFAULT_OUTPUT = Path("outputs/persistence-judge-rerun")
PROMPT_PATH = DEFAULT_RELEASE / "prompts/persistence-eval-prompts.yaml"
SYSTEM_PROMPT = "You are a mutation persistence evaluator."
OPENAI_MODEL = "gpt-4o-mini-2024-07-18"
LOCAL_MODELS = {
    "qwen": "qwen3.8:27b",
    "gemma": "gemma4:31b",
}
EXPECTED_ROWS = 132
EXPECTED_DECISIONS = 264


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_bool(value: object) -> bool:
    normalized = str(value).strip().upper()
    if normalized == "TRUE":
        return True
    if normalized == "FALSE":
        return False
    raise ValueError(f"Invalid human label: {value!r}")


def prompt_key(mutator: str) -> str:
    if not re.fullmatch(r"(?:ea|mm)-[a-z-]+", mutator):
        raise ValueError(f"Unexpected mutator name: {mutator!r}")
    return f"{mutator[3:]}-persistence-eval"


def freeze_and_expand_source(output_dir: Path) -> tuple[Path, pd.DataFrame]:
    source_dir = output_dir / "source"
    source_dir.mkdir(parents=True, exist_ok=True)
    frozen = source_dir / SOURCE_PATH.name
    if not frozen.exists():
        shutil.copy2(SOURCE_PATH, frozen)
    if sha256_file(frozen) != sha256_file(SOURCE_PATH):
        raise RuntimeError("Frozen persistence-label source differs from canonical source")

    source = pd.read_csv(frozen)
    if len(source) != EXPECTED_ROWS:
        raise RuntimeError(f"Expected {EXPECTED_ROWS} source rows, found {len(source)}")

    decisions: list[dict] = []
    for chain_id, row in source.iterrows():
        decisions.extend(
            [
                {
                    "decision_id": 2 * int(chain_id),
                    "chain_id": int(chain_id),
                    "position": "M1",
                    "mutator": str(row["Mutator 1"]),
                    "original_prompt": str(row["Input Prompt"]),
                    "final_prompt": str(row["Jailbreak Prompt 2"]),
                    "human_label": parse_bool(row["M1 Persistence GT"]),
                },
                {
                    "decision_id": 2 * int(chain_id) + 1,
                    "chain_id": int(chain_id),
                    "position": "M2",
                    "mutator": str(row["Mutator 2"]),
                    "original_prompt": str(row["Jailbreak Prompt 1"]),
                    "final_prompt": str(row["Jailbreak Prompt 2"]),
                    "human_label": parse_bool(row["M2 Persistence GT"]),
                },
            ]
        )
    expanded = pd.DataFrame(decisions)
    if len(expanded) != EXPECTED_DECISIONS or expanded["decision_id"].duplicated().any():
        raise RuntimeError("Persistence decision expansion failed its cardinality check")
    return frozen, expanded


def load_templates(decisions: pd.DataFrame) -> dict[str, str]:
    templates = yaml.safe_load(PROMPT_PATH.read_text(encoding="utf-8"))
    needed = {prompt_key(value) for value in decisions["mutator"].unique()}
    missing = needed.difference(templates)
    if missing:
        raise RuntimeError(f"Missing persistence prompt definitions: {sorted(missing)}")
    return {key: templates[key] for key in sorted(needed)}


def normalize_label(raw: str) -> str | None:
    match = re.fullmatch(r"\s*(TRUE|FALSE)\s*[.!]?\s*", raw, flags=re.IGNORECASE)
    return match.group(1).upper() if match else None


def init_db(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path)
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS evaluations (
            decision_id INTEGER PRIMARY KEY,
            raw_output TEXT,
            prediction TEXT,
            attempts INTEGER NOT NULL,
            error TEXT,
            started_at TEXT NOT NULL,
            finished_at TEXT NOT NULL
        )
        """
    )
    connection.commit()
    return connection


def openai_caller(api_key: str) -> Callable[[str], str]:
    local = threading.local()

    def call(user_prompt: str) -> str:
        if not hasattr(local, "client"):
            local.client = OpenAI(api_key=api_key, timeout=120, max_retries=0)
        completion = local.client.chat.completions.create(
            model=OPENAI_MODEL,
            temperature=0,
            max_tokens=8,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
        )
        return completion.choices[0].message.content or ""

    return call


def ollama_caller(model: str, urls: list[str]) -> Callable[[str], str]:
    counter = iter(range(10**12))
    lock = threading.Lock()

    def call(user_prompt: str) -> str:
        with lock:
            index = next(counter)
        url = urls[index % len(urls)].rstrip("/") + "/api/chat"
        response = requests.post(
            url,
            json={
                "model": model,
                "stream": False,
                "think": False,
                "keep_alive": "30m",
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                "options": {
                    "temperature": 0,
                    "num_predict": 8,
                    "num_ctx": 8192,
                    "seed": 20260914,
                },
            },
            timeout=600,
        )
        response.raise_for_status()
        return response.json()["message"]["content"]

    return call


def evaluate_decision(
    decision: dict,
    templates: dict[str, str],
    caller: Callable[[str], str],
    max_attempts: int,
) -> dict:
    started = utc_now()
    last_raw = ""
    last_error = ""
    for attempt in range(1, max_attempts + 1):
        try:
            template = templates[prompt_key(decision["mutator"])]
            user_prompt = template.format(
                original_prompt=decision["original_prompt"],
                final_prompt=decision["final_prompt"],
            )
            last_raw = caller(user_prompt)
            prediction = normalize_label(last_raw)
            if prediction is None:
                raise ValueError(f"Invalid judge output: {last_raw[:120]!r}")
            return {
                "decision_id": decision["decision_id"],
                "raw_output": last_raw,
                "prediction": prediction,
                "attempts": attempt,
                "error": None,
                "started_at": started,
                "finished_at": utc_now(),
            }
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            if attempt < max_attempts:
                time.sleep(min(2**attempt, 20))
    return {
        "decision_id": decision["decision_id"],
        "raw_output": last_raw,
        "prediction": None,
        "attempts": max_attempts,
        "error": last_error[:1000],
        "started_at": started,
        "finished_at": utc_now(),
    }


def export_predictions(
    connection: sqlite3.Connection,
    decisions: pd.DataFrame,
    judge: str,
    output_dir: Path,
) -> Path:
    evaluated = pd.read_sql_query(
        "SELECT decision_id, raw_output, prediction, attempts, error, "
        "started_at, finished_at FROM evaluations ORDER BY decision_id",
        connection,
    )
    metadata = decisions[[
        "decision_id", "chain_id", "position", "mutator", "human_label"
    ]]
    exported = metadata.merge(evaluated, on="decision_id", how="left")
    exported.insert(1, "judge", judge)
    path = output_dir / f"predictions_{judge}.csv"
    exported.to_csv(path, index=False)
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--judge", choices=["gpt4o_mini", "qwen", "gemma"], required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-attempts", type=int, default=4)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--api-key-file",
        type=Path,
        default=None,
    )
    parser.add_argument("--ollama-url", action="append", default=[])
    parser.add_argument("--execute", action="store_true", help="Explicitly allow paid/local inference")
    args = parser.parse_args()

    check_manifest(DEFAULT_RELEASE)
    check_support()
    args.output_dir = output_directory(args.output_dir)
    lock_run(args.output_dir, {"kind": "persistence", "judge": args.judge, "limit": args.limit,
                              "source_sha256": sha256_file(SOURCE_PATH), "template_sha256": sha256_file(PROMPT_PATH),
                              "temperature": 0, "local_seed": 20260914})
    frozen_source, decisions = freeze_and_expand_source(args.output_dir)
    if args.limit is not None:
        decisions = decisions.iloc[: args.limit].copy()
    templates = load_templates(decisions)
    if not args.execute:
        print(f"Initialized {len(decisions)} decisions. Add --execute to call models.")
        return

    if args.judge == "gpt4o_mini":
        caller = openai_caller(load_key(args.api_key_file))
        model = OPENAI_MODEL
        endpoints: list[str] = []
    else:
        if not args.ollama_url:
            raise RuntimeError("At least one --ollama-url is required for a local judge")
        model = LOCAL_MODELS[args.judge]
        endpoints = args.ollama_url
        caller = ollama_caller(model, endpoints)

    db_path = args.output_dir / f"checkpoint_{args.judge}.sqlite3"
    connection = init_db(db_path)
    complete = {
        row[0]
        for row in connection.execute(
            "SELECT decision_id FROM evaluations WHERE prediction IS NOT NULL"
        ).fetchall()
    }
    pending = [
        row._asdict()
        for row in decisions.itertuples(index=False)
        if int(row.decision_id) not in complete
    ]
    print(
        f"judge={args.judge} model={model} total={len(decisions)} "
        f"complete={len(complete)} pending={len(pending)} workers={args.workers}",
        flush=True,
    )

    started_at = utc_now()
    processed = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                evaluate_decision, decision, templates, caller, args.max_attempts
            ): decision["decision_id"]
            for decision in pending
        }
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            connection.execute(
                """
                INSERT INTO evaluations
                    (decision_id, raw_output, prediction, attempts, error,
                     started_at, finished_at)
                VALUES
                    (:decision_id, :raw_output, :prediction, :attempts, :error,
                     :started_at, :finished_at)
                ON CONFLICT(decision_id) DO UPDATE SET
                    raw_output=excluded.raw_output,
                    prediction=excluded.prediction,
                    attempts=excluded.attempts,
                    error=excluded.error,
                    started_at=excluded.started_at,
                    finished_at=excluded.finished_at
                """,
                result,
            )
            connection.commit()
            processed += 1
            if processed % 25 == 0 or processed == len(pending):
                success = connection.execute(
                    "SELECT COUNT(*) FROM evaluations WHERE prediction IS NOT NULL"
                ).fetchone()[0]
                errors = connection.execute(
                    "SELECT COUNT(*) FROM evaluations WHERE prediction IS NULL"
                ).fetchone()[0]
                print(
                    f"judge={args.judge} processed={processed}/{len(pending)} "
                    f"saved_success={success} saved_errors={errors}",
                    flush=True,
                )

    prediction_path = export_predictions(connection, decisions, args.judge, args.output_dir)
    successful = connection.execute(
        "SELECT COUNT(*) FROM evaluations WHERE prediction IS NOT NULL"
    ).fetchone()[0]
    failed = connection.execute(
        "SELECT COUNT(*) FROM evaluations WHERE prediction IS NULL"
    ).fetchone()[0]
    used_prompt_hashes = {
        key: sha256_bytes(value.encode("utf-8")) for key, value in templates.items()
    }
    manifest = {
        "status": "PASS" if successful == len(decisions) and failed == 0 else "PARTIAL",
        "judge": args.judge,
        "model": model,
        "source_rows": EXPECTED_ROWS,
        "dataset_decisions": len(decisions),
        "successful": successful,
        "failed": failed,
        "source_path": str(SOURCE_PATH),
        "source_sha256": sha256_file(frozen_source),
        "prompt_path": str(PROMPT_PATH),
        "prompt_file_sha256": sha256_file(PROMPT_PATH),
        "used_prompt_sha256": used_prompt_hashes,
        "system_prompt_sha256": sha256_bytes(SYSTEM_PROMPT.encode("utf-8")),
        "comparison_mapping": {
            "M1": "Input Prompt -> Jailbreak Prompt 2",
            "M2": "Jailbreak Prompt 1 -> Jailbreak Prompt 2",
        },
        "temperature": 0,
        "local_seed": 20260914 if args.judge != "gpt4o_mini" else None,
        "thinking": False if args.judge != "gpt4o_mini" else None,
        "endpoints": endpoints,
        "started_at": started_at,
        "finished_at": utc_now(),
        "prediction_sha256": sha256_file(prediction_path),
        "checkpoint_sha256": sha256_file(db_path),
    }
    (args.output_dir / f"manifest_{args.judge}.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2), flush=True)
    connection.close()
    if failed or successful != len(decisions):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
