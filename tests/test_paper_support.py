"""Offline tests for the restored evidence and safe rerun entry points."""
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import experiment as core
import run_intent_judge_validation as intent
import run_persistence_judge_validation as persistence
import run_mutator_alignment_pilot as pilot
from release_inputs import DEFAULT_RELEASE, initialize_store, load_inputs
from paper_support import SUPPORT, check_support, lock_run, output_directory
from run_frozen_experiment import attach_archived_responses
from analyze_intent_judge_validation import cluster_bootstrap, metrics
from analyze_target_run import analyze


class PaperSupportTests(unittest.TestCase):
    def test_support_manifest(self):
        self.assertEqual(len(check_support()), 64)

    def test_support_tampering(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            (path / "a").write_text("wrong")
            (path / "manifest.json").write_text(json.dumps({"files": {"a": {
                "sha256": hashlib.sha256(b"right").hexdigest(), "bytes": 5}}}))
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                check_support(path)

    def test_published_directories_protected(self):
        for directory in [ROOT, DEFAULT_RELEASE, SUPPORT, ROOT / "results", ROOT / "scripts"]:
            with self.subTest(directory=directory), self.assertRaises(ValueError):
                output_directory(directory)

    def test_changed_run_settings_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            lock_run(tmp, {"judge": "a"})
            lock_run(tmp, {"judge": "a"})
            with self.assertRaisesRegex(ValueError, "settings changed"):
                lock_run(tmp, {"judge": "b"})

    def test_validation_initialization_never_downloads_or_calls_models(self):
        for module in [intent, persistence]:
            with self.subTest(module=module.__name__), tempfile.TemporaryDirectory() as tmp, \
                 patch.object(sys, "argv", [module.__name__, "--judge", "gpt4o_mini", "--output-dir", tmp]), \
                 patch.object(module, "openai_caller", side_effect=AssertionError("API call")), \
                 patch.object(module.requests, "get", side_effect=AssertionError("Download")), \
                 patch.object(module.requests, "post", side_effect=AssertionError("Model call")), \
                 patch.object(module, "load_key", side_effect=AssertionError("Credential access")), \
                 redirect_stdout(io.StringIO()):
                module.main()
                self.assertTrue((Path(tmp) / "run_identity.json").exists())

    def test_persistence_human_mapping_matches_released_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, expanded = persistence.freeze_and_expand_source(Path(tmp))
        labels = pd.read_csv(DEFAULT_RELEASE / "validation/persistence/predictions_gpt4o_mini.csv")
        self.assertEqual(len(expanded), 264)
        for key in ["decision_id", "chain_id", "position", "mutator", "human_label"]:
            self.assertEqual(expanded[key].tolist(), labels[key].tolist())

    def test_intent_cluster_metadata(self):
        source = pd.read_csv(SUPPORT / "validation/intent_source_metadata.csv")
        labels = pd.read_csv(DEFAULT_RELEASE / "validation/intent/predictions_gpt4o_mini.csv")
        self.assertEqual(len(source), 1361)
        self.assertEqual(source.cluster_id.nunique(), 39)
        for key in ["row_id", "model", "jailbreak", "label_median"]:
            self.assertEqual(source[key].tolist(), labels[key].tolist())

    def test_bootstrap_is_reproducible(self):
        frame = pd.DataFrame({"forbidden_prompt": [0, 0, 1, 1],
                              "truth": [True, False, False, True],
                              "prediction": [True, True, False, True]})
        self.assertEqual(cluster_bootstrap(frame, "truth", "prediction", 100),
                         cluster_bootstrap(frame, "truth", "prediction", 100))
        self.assertEqual(metrics(frame.truth.to_numpy(), frame.prediction.to_numpy())["accuracy"], 0.75)

    def test_pilot_initialization_has_no_model_calls(self):
        with tempfile.TemporaryDirectory() as tmp, \
             patch.object(sys, "argv", ["pilot", "--run-dir", tmp]), \
             patch.object(pilot, "read_shared_api_key", side_effect=AssertionError("Credentials")), \
             patch.object(core, "run_tasks", side_effect=AssertionError("Inference")), \
             redirect_stdout(io.StringIO()):
            self.assertEqual(pilot.main(), 0)
            store = core.Store(Path(tmp) / "state.sqlite3")
            self.assertEqual(sum(store.counts()["chains"].values()), 2640)
            saved = json.loads(store.conn.execute("SELECT value FROM metadata WHERE key='sample_prompt_rows'").fetchone()[0])
            expected = json.loads((SUPPORT / "writer_pilot/settings.json").read_text())["sample_prompt_rows"]
            self.assertEqual(saved, expected)
            store.conn.close()

    def test_rejudging_uses_only_matching_archived_responses(self):
        base, chains, _ = load_inputs(DEFAULT_RELEASE, "r1", "stability", max_prompts=1)
        with tempfile.TemporaryDirectory() as tmp:
            store = core.Store(Path(tmp) / "state.sqlite3")
            initialize_store(store, base, chains, {"rejudge": True})
            attach_archived_responses(store, DEFAULT_RELEASE, "r1", "stability", 1, base, chains)
            for table in ["baselines", "chains"]:
                self.assertEqual(store.conn.execute(f"SELECT COUNT(*) FROM {table} WHERE target_response IS NULL").fetchone()[0], 0)
                self.assertEqual(store.conn.execute(f"SELECT COUNT(*) FROM {table} WHERE safety_eval IS NOT NULL").fetchone()[0], 0)
            store.conn.close()

    def test_new_run_analysis_and_missing_component_guard(self):
        base, chains, _ = load_inputs(DEFAULT_RELEASE, "r1", "stability", max_prompts=1)
        with tempfile.TemporaryDirectory() as tmp:
            store = core.Store(Path(tmp) / "state.sqlite3")
            initialize_store(store, base, chains, {"target": "r1", "scope": "stability"})
            for table in ["baselines", "chains"]:
                store.conn.execute(f"UPDATE {table} SET status='done', safety_eval='Safe', intent_eval='FALSE'")
            store.conn.commit()
            with redirect_stdout(io.StringIO()):
                summary = analyze(tmp, Path(tmp) / "analysis")
            self.assertEqual(summary["tested_pairs"], 7)
            self.assertFalse(summary["all_520"])
            self.assertEqual(summary["pair_successes"], 0)
            store.conn.execute("DELETE FROM baselines WHERE task_id=?", (base.iloc[0].task_id,))
            store.conn.commit()
            with self.assertRaisesRegex(ValueError, "matching standalone"):
                analyze(tmp, Path(tmp) / "missing")
            store.conn.close()


if __name__ == "__main__":
    unittest.main()
