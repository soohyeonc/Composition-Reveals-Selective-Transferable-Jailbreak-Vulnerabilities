# Composition Reveals Selective, Transferable Jailbreak Vulnerabilities in Four Large Language Models

Code, frozen inputs, recorded outputs, and analysis for the four-target study of
compositional jailbreaking. The experiments compare twelve individual prompt
transformations with all 132 ordered pairs on 520 AdvBench requests. One frozen
GPT-3.5 mutation batch is shared across GPT-5.6 Luna, DeepSeek R1 8B, Gemma 4 31B,
and Qwen3.8 27B.

The repository includes completed discovery matrices, **seven stability
conditions per target**, five repetitions, historical completeness measurements,
the mutation-writer comparison, and both three-judge validation studies. It does
not include unreported stability conditions or experiments on older targets.

**Artifact scope:** all included empirical artifacts pass offline reproduction.
Verification covers the distributed datasets and analyses, not every manuscript
statement or the independent correctness of automated judgments. See
[the paper-to-artifact map](REPRODUCIBILITY.md) for the precise boundary.

**Research content warning:** inputs and recorded model responses contain harmful
requests and jailbreak material. Use these artifacts for controlled security
evaluation. No model weights or credentials are distributed.

## Reproduce the recorded results

From the repository root:

```bash
conda env create -f environment.yml
conda activate compositional_jailbreaking
python scripts/reproduce_paper.py
```

If the environment already exists, activate it and ensure its dependencies match
`environment.yml`. Do not recreate an active research environment or install into
Conda's base environment.

This command requires no API key, GPU, download, or model call. It recomputes the
results into `outputs/paper/` and checks them against the immutable references.
It verifies:

- Four full target matrices, 528 ordered-pair comparisons, and equal denominators.
- The 28 reported stability conditions and all 140 pair-runs.
- All 442,520 released target-response records against their original hashes.
- Four mutation batches, the normalized completeness gate, and positional effects.
- Intent and persistence metrics, disagreements, and 10,000-resample clustered
  bootstrap confidence intervals.
- The writer-pilot statistics and its 2,000-resample bootstrap analysis.

The command also regenerates nine empirical figure panels and copies the
conceptual pipeline illustration. It reports `PASS` for included evidence and
records the artifact's scope separately in `coverage.json`. Use `--no-figures`
for numerical checks only. The optional `--require-complete` flag additionally
requires whole-paper coverage, which is broader than this package's scope.

The smaller original audit remains available:

```bash
python scripts/verify_tifs_release.py
python -m unittest discover -s tests -v
```

Tests use local data and mocked responses. They do not spend API credits or load
models onto a GPU.

## Read the results

Start with [FOUR_MODEL_RESULTS.md](FOUR_MODEL_RESULTS.md). The complete,
paper-aligned outputs are already saved under [results/paper/](results/paper/).
No rerun is needed to inspect them.

| Results | File |
| --- | --- |
| Model-level summary | [Table VI](results/paper/table06_model_summary.csv) |
| Every individual mutator | [Individual results](results/paper/individual_mutator_results.csv) |
| All 528 discovery pair comparisons | [Pair results](results/paper/all_discovery_pairs.csv) |
| Featured discovery examples | [Table VII](results/paper/table07_featured_discovery.csv) |
| Featured five-run summaries | [Table VIII](results/paper/table08_featured_stability.csv) |
| Featured per-run counts | [Table IX](results/paper/table09_featured_runs.csv) |
| All seven conditions, original and runs 1–5 | [Readable stability tables](results/paper/STABILITY_TABLES.md) |
| All 28 stability summaries | [Supplemental table](results/paper/table11_all_stability_conditions.csv) |
| Completeness and writer comparison | [Table V](results/paper/table05_completeness_batches.csv), [Table X](results/paper/table10_writer_pilot.csv) |
| Judge validation | [Intent](results/paper/validation/intent/SUMMARY.md), [persistence](results/paper/validation/persistence/SUMMARY.md) |
| Figures and claim coverage | [Figures](results/paper/figures/), [coverage report](results/paper/coverage.json) |

All reported data come from the completed dataset frozen on 14 September 2026.
The seven-condition panels were selected retrospectively. Their five-run outcomes
describe those conditions, not general consistency across all 132 pairs. The
four historical mutation batches are not controlled repetitions of a pinned
writer configuration.

## Run a fresh target evaluation

The public runner uses the released frozen prompts and persistence labels. It
does not regenerate mutations or require access to GPT-3.5.

```bash
python scripts/run_frozen_experiment.py \
  --target r1 --scope stability --run 1 \
  --output-dir outputs/r1-run1
```

Without `--execute`, this initializes a checkpoint only. Targets are `luna`,
`r1`, `gemma`, and `qwen`. Discovery uses all twelve mutators and 132 pairs.
Stability uses the target's seven reported pairs and the recorded settings for
run 1 through 5. Component baselines are deduplicated within each run.

To execute, set `OPENAI_API_KEY` securely, make the target available, and add
`--execute`:

```bash
python scripts/run_frozen_experiment.py \
  --target r1 --scope stability --run 1 \
  --output-dir outputs/r1-run1 \
  --ollama-url http://127.0.0.1:11434 \
  --budget-usd 25 --execute

python scripts/analyze_target_run.py --run-dir outputs/r1-run1
```

Local targets require Ollama with the recorded checkpoint digest. The runner
checks that digest before inference. Luna requires access to its recorded hosted
model. All targets use the recorded paid OpenAI judge. Historical hosted models
are not guaranteed to remain available.

`--budget-usd` is an estimated stopping threshold using historical prices, not a
guaranteed billing cap. Check current provider pricing and account-level limits
before execution. For a small initialization test, use `--max-prompts 1` and a
separate output directory. That subset is not an all-520 reproduction.

The runner checkpoints every completed API stage. Repeat the command to resume;
changed inputs or settings require a new directory. Incomplete execution returns
a nonzero status. New outputs never replace published measurements. Sampling,
hardware, server versions, and provider changes can affect fresh results.

## Audit the recorded judgments and writer comparison

To rejudge archived responses without running the target again, add
`--rejudge-archived` to the frozen-input runner:

```bash
python scripts/run_frozen_experiment.py \
  --target qwen --scope stability --run 1 --rejudge-archived \
  --output-dir outputs/qwen-rejudge
```

Adding `--execute` makes safety and intent judge calls only. Provider-blocked
trials remain failed attacks. Use `analyze_target_run.py` to analyze new
judgments separately from canonical results.

The other rerun entry points also require explicit `--execute` for model calls:

```bash
python scripts/run_intent_judge_validation.py \
  --judge gpt4o_mini --output-dir outputs/intent-rerun
python scripts/run_persistence_judge_validation.py \
  --judge gpt4o_mini --output-dir outputs/persistence-rerun
python scripts/run_mutator_alignment_pilot.py --run-dir outputs/writer-rerun
```

Judge choices are `gpt4o_mini`, `qwen`, and `gemma`. Local judges need an
`--ollama-url`. Intent-judge inference downloads the original StrongREJECT corpus
and checks its recorded SHA-256 hash. Its source texts are not redistributed
here; all labels and opaque prompt-cluster IDs needed for offline numerical
reproduction are included. The writer pilot uses the same twenty selected
prompt identifiers and 132 pairs, with no target-model evaluation.

See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for individual analysis commands,
schemas, settings, and limitations.

## Repository structure

```text
FOUR_MODEL_RESULTS.md       Living summary of canonical four-target results
REPRODUCIBILITY.md          Paper-to-data map and reproduction boundaries
environment.yml            Pinned analysis and inference dependencies
scripts/                   Offline analyses, figures, and safe rerun commands
tests/                     Offline regression and workflow tests
release/tifs-20261005/      Original immutable labels, prompts, settings, hashes
release/paper-support/     Hash-pinned responses and supporting source records
results/paper/             Recomputed, checked paper-aligned tables and figures
```

The original 92-file release remains unchanged. The support dataset has its own
manifest and was validated against original response hashes and canonical
metrics. Its approximately 324 MB of compressed evidence is retained for
auditability. Unreported experiment databases, private logs, credentials, and
archived notebooks are excluded.

## License and attribution

Original project code is available under the [MIT License](LICENSE). Third-party
datasets and model outputs are not relicensed by that license. See
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for attribution and source terms.
