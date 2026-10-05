"""Offline tests. Model transport is always mocked; no credentials are needed."""
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from contextlib import redirect_stdout, redirect_stderr
from unittest.mock import patch, Mock

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import experiment as core
from ollama_backend import OllamaExperiment
from release_inputs import (DEFAULT_RELEASE, MODELS, check_manifest,
                            initialize_store, load_inputs, select_settings)
from run_frozen_experiment import check_ollama, main
from plot_results import matrix


class ReproductionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base, cls.chains, cls.record = load_inputs(DEFAULT_RELEASE, "r1", "stability")

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.directory = Path(self.temp.name)
        self.store = core.Store(self.directory / "state.sqlite3")

    def tearDown(self):
        self.store.conn.close()
        self.temp.cleanup()

    def initialize(self):
        initialize_store(self.store, self.base.iloc[:1], self.chains.iloc[:1], {"test": 1})

    def mocked_experiment(self):
        experiment = core.Experiment(
            self.store, "not-a-real-key", {},
            core.load_yaml(DEFAULT_RELEASE / "prompts/evaluator-system-prompts.yaml"),
            {}, core.Budget(0, 25), core.RateLimiter(300))
        usage = core.Usage("mock", "mock", 0, 0, 0, 0, 0, None, None)
        def completion(**kwargs):
            if kwargs["model"] == core.TARGET_MODEL:
                return "Harmless mock response", usage
            if kwargs["system_prompt"] == core.SAFETY_SYSTEM_PROMPT:
                return "Unsafe", usage
            if kwargs["system_prompt"] == core.INTENT_SYSTEM_PROMPT:
                return "TRUE", usage
            self.fail("Unexpected mutation or persistence call")
        experiment.completion = Mock(side_effect=completion)
        return experiment

    def test_manifest(self):
        self.assertEqual(len(check_manifest(DEFAULT_RELEASE)), 64)

    def test_manifest_tampering_fails(self):
        fixture = self.directory / "release"
        fixture.mkdir()
        (fixture / "data.txt").write_text("changed")
        (fixture / "manifest.json").write_text(json.dumps({"files": {
            "data.txt": hashlib.sha256(b"original").hexdigest()}}))
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            check_manifest(fixture)

    def test_all_stability_panels(self):
        for target in MODELS:
            with self.subTest(target=target):
                base, chains, record = load_inputs(DEFAULT_RELEASE, target, "stability")
                self.assertEqual(len(chains), 7 * 520)
                self.assertEqual(base.groupby("mutator").size().unique().tolist(), [520])
                self.assertEqual(record["phase"], "stability")
                self.assertFalse(chains[["m1_persistence", "m2_persistence"]].isna().any().any())

    def test_all_discovery_inputs(self):
        base, chains, _ = load_inputs(DEFAULT_RELEASE, "luna", "discovery")
        self.assertEqual((len(base), len(chains)), (6240, 68640))
        self.assertEqual(chains.groupby(["mutator_1", "mutator_2"]).ngroups, 132)
        self.assertEqual(int(chains.jailbreak_prompt_1.eq("").sum()), 1)
        self.assertFalse(chains.jailbreak_prompt_2.eq("").any())

    def test_every_recorded_repetition(self):
        for target, seed_root in [("qwen", 91000), ("r1", 92000), ("gemma", 93000)]:
            for run in range(1, 6):
                self.assertEqual(select_settings(DEFAULT_RELEASE, target, "stability", run)
                                 ["settings"]["target_seed"], seed_root + run)

    def test_invalid_subset_rejected(self):
        for size in [0, -1, 521]:
            with self.assertRaises(ValueError):
                load_inputs(DEFAULT_RELEASE, "r1", max_prompts=size)

    def test_original_task_ids_and_seeds(self):
        row = self.chains.iloc[0]
        identity = core.stable_id("chain", row.prompt_row, row.mutator_1, row.mutator_2)
        self.assertEqual(row.task_id, identity)
        experiment = OllamaExperiment(self.store, "mock", {}, {}, {}, core.Budget(0, 1),
                                     core.RateLimiter(300), ollama_urls=["http://unused"],
                                     target_model="deepseek-r1:8b", target_timeout=1,
                                     think=False, num_ctx=8192, target_seed=92001)
        expected = int.from_bytes(hashlib.sha256(f"92001\x1fchain\x1f{identity}".encode())
                                  .digest()[:4], "big")
        self.assertEqual(experiment.task_seed("chain", identity), expected)

    def test_resume_and_frozen_fields(self):
        self.initialize()
        experiment = self.mocked_experiment()
        experiment.process_baseline(self.base.iloc[0].task_id)
        experiment.process_chain(self.chains.iloc[0].task_id)
        self.assertEqual(experiment.completion.call_count, 6)
        self.initialize()
        experiment.process_baseline(self.base.iloc[0].task_id)
        experiment.process_chain(self.chains.iloc[0].task_id)
        self.assertEqual(experiment.completion.call_count, 6)
        self.assertEqual(self.store.counts(), {"baselines": {"done": 1}, "chains": {"done": 1}})
        with self.assertRaisesRegex(RuntimeError, "metadata mismatch"):
            initialize_store(self.store, self.base.iloc[:1], self.chains.iloc[:1], {"test": 2})

    def test_missing_frozen_prompt_never_regenerated(self):
        self.initialize()
        self.store.conn.execute("UPDATE baselines SET mutated_prompt=NULL")
        self.store.conn.commit()
        experiment = self.mocked_experiment()
        with self.assertRaisesRegex(ValueError, "Missing frozen"):
            experiment.process_baseline(self.base.iloc[0].task_id)
        experiment.completion.assert_not_called()

    def test_provider_blocks_are_failed_attacks(self):
        self.initialize()
        task = self.base.iloc[0].task_id
        self.store.mark_provider_block("baselines", task, "bio_policy")
        experiment = self.mocked_experiment()
        experiment.process_baseline(task)
        experiment.completion.assert_not_called()
        row = self.store.row("baselines", task)
        self.assertEqual(row["status"], "done")
        self.assertEqual(row["intent_eval"].lower(), "false")
        self.assertEqual(row["safety_eval"], "Safe")

    def test_ollama_checkpoint_check(self):
        settings = {"target_model": "deepseek-r1:8b", "target_model_digest": "abc123"}
        response = Mock()
        response.json.return_value = {"models": [{"name": "deepseek-r1:8b", "digest": "abc123456"}]}
        with patch("run_frozen_experiment.requests.get", return_value=response):
            check_ollama(["http://unused"], settings)
            response.json.return_value["models"][0]["digest"] = "different"
            with self.assertRaises(ValueError):
                check_ollama(["http://unused"], settings)

    def test_hosted_request_settings(self):
        experiment = self.mocked_experiment()
        experiment.completion = core.Experiment.completion.__get__(experiment)
        client = Mock()
        client.chat.completions.create.return_value = SimpleNamespace(
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5),
            choices=[SimpleNamespace(message=SimpleNamespace(content="mock"))], model="mock")
        experiment.client = Mock(return_value=client)
        with patch.object(core, "TARGET_MODEL", "gpt-5.6-luna"):
            experiment.completion(model="gpt-5.6-luna", system_prompt=core.TARGET_SYSTEM_PROMPT,
                                  user_prompt="test", max_tokens=4096, reasoning_effort="none")
        kwargs = client.chat.completions.create.call_args.kwargs
        self.assertEqual(kwargs["max_completion_tokens"], 4096)
        self.assertEqual(kwargs["reasoning_effort"], "none")
        self.assertNotIn("temperature", kwargs)

    def test_ollama_request_settings(self):
        experiment = OllamaExperiment(self.store, "mock", {}, {}, {}, core.Budget(0, 1),
                                     core.RateLimiter(300), ollama_urls=["http://unused"],
                                     target_model="deepseek-r1:8b", target_timeout=1,
                                     think=False, num_ctx=8192, target_seed=92001)
        session = Mock()
        session.post.return_value.json.return_value = {"message": {"content": "mock"}}
        experiment.ollama_session = Mock(return_value=session)
        experiment.completion(model="deepseek-r1:8b", system_prompt=core.TARGET_SYSTEM_PROMPT,
                              user_prompt="test", max_tokens=4096)
        payload = session.post.call_args.kwargs["json"]
        self.assertFalse(payload["think"])
        self.assertEqual(payload["options"], {"num_predict": 4096, "num_ctx": 8192, "seed": 92001})

    def test_cli_mock_execution_and_failure_status(self):
        usage = core.Usage("mock", "mock", 0, 0, 0, 0, 0, None, None)
        def mock_completion(**kwargs):
            text = "mock"
            if kwargs["system_prompt"] == core.SAFETY_SYSTEM_PROMPT:
                text = "Safe"
            elif kwargs["system_prompt"] == core.INTENT_SYSTEM_PROMPT:
                text = "FALSE"
            return text, usage
        for fail in [False, True]:
            output = self.directory / ("failed" if fail else "finished")
            args = ["run_frozen_experiment.py", "--target", "luna", "--max-prompts", "1",
                    "--output-dir", str(output), "--execute"]
            side_effect = RuntimeError("Mock provider failure") if fail else mock_completion
            with patch.object(sys, "argv", args), \
                 patch("experiment.read_api_key", return_value="mock"), \
                 patch("experiment.Experiment.completion", side_effect=side_effect), \
                 redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                self.assertEqual(main(), 2 if fail else 0)
            summary = json.loads((output / "summary.json").read_text())
            self.assertEqual(summary["complete"], not fail)

    def test_cli_default_has_no_network_or_credentials(self):
        args = ["run_frozen_experiment.py", "--target", "qwen", "--max-prompts", "1",
                "--output-dir", str(self.directory / "cli")]
        with patch.object(sys, "argv", args), \
             patch("run_frozen_experiment.requests.get", side_effect=AssertionError("Network")), \
             patch("experiment.read_api_key", side_effect=AssertionError("Credentials")), \
             redirect_stdout(io.StringIO()):
            self.assertEqual(main(), 0)

    def test_release_cannot_be_output(self):
        args = ["run_frozen_experiment.py", "--target", "r1", "--output-dir", str(DEFAULT_RELEASE)]
        with patch.object(sys, "argv", args), self.assertRaisesRegex(ValueError, "immutable"):
            main()

    def test_plot_masks_match_paper(self):
        pairs = pd.read_csv(DEFAULT_RELEASE / "discovery/pair_metrics.csv")
        for slug, expected in [("luna", 13), ("r1", 34), ("gemma", 16), ("qwen", 9)]:
            frame = pairs.loc[pairs.model.eq(MODELS[slug])].rename(columns={
                "mutator_1": "m1", "mutator_2": "m2", "matched_prompts": "complete_count"})
            display = matrix(frame, "chain_successes", frame.average_gate_matched_success)
            self.assertEqual(int(display.notna().sum().sum()), expected)


if __name__ == "__main__":
    unittest.main()
