# Paper-to-artifact coverage

This package distinguishes recomputing recorded measurements from obtaining
identical outputs in a fresh stochastic model run. The former is checked
offline. The latter cannot be guaranteed, particularly for hosted models.

The reported empirical scope is four target models and seven stability
conditions per target, each with five repetitions. GPT-3.5 is the writer, not a
fifth target. The GPT-5.6 writer comparison evaluates persistence, not target ASR.

## Paper map

All paths below are relative to this repository. Run
`python scripts/reproduce_paper.py` to regenerate the computed outputs into
`outputs/paper/`. Checked copies are already published under `results/paper/`.

| Paper item | Inputs | Checked output |
| --- | --- | --- |
| Table I, mutator taxonomy and illustrations | `release/tifs-20261005/prompts/*yaml` and frozen prompts | Descriptive definitions, not an estimated result |
| Table II, target settings | `release/tifs-20261005/run_settings.json` | `results/paper/table02_recorded_settings.csv` |
| Table III, intent validation | `validation/intent/predictions_*.csv` in the original release; `paper-support/validation/intent_source_metadata.csv` | `results/paper/table03_intent_validation.csv` |
| Table IV, persistence validation | `validation/persistence/predictions_*.csv` and source human annotations | `results/paper/table04_persistence_validation.csv` |
| Table V, four-batch completeness | Three support batch exports plus discovery batch D labels | `results/paper/table05_completeness_batches.csv` |
| Table VI, full model matrices | `release/tifs-20261005/discovery/*csv*` | `results/paper/table06_model_summary.csv` |
| Table VII, featured discovery pairs | Discovery labels joined to standalone labels on prompt and mutator | `results/paper/table07_featured_discovery.csv` |
| Tables VIII–IX, featured stability | Original release `stability/*csv.gz` and `panel.json` | `results/paper/table08_featured_stability.csv`, `table09_featured_runs.csv` |
| Table X, writer comparison | Original `writer_pilot/row_labels.csv` plus support candidate generations | `results/paper/table10_writer_pilot.csv` |
| Supplemental Table XI, complete reported panel | Same seven-condition stability data | `results/paper/table11_all_stability_conditions.csv`, `STABILITY_TABLES.md` |
| Figure 1, conceptual pipeline | Preserved authored illustration | `results/paper/figures/chained-mutator.png` |
| Figure 2, judge precision/recall | Recomputed validation metrics | `results/paper/figures/judge_precision_recall.png` |
| Figure 3, completeness panels | Normalized four-batch pair metrics | `results/paper/figures/fig3*.png` |
| Figure 4, four target screens | Matched prompt-level comparisons and mean completeness gate | `results/paper/figures/fig4*.png` |
| Figure 5, Jaccard overlap | Raw-positive pair sets from complete discovery matrices | `results/paper/figures/fig_jaccard_dotplot.png` |
| Prose, positional effects, first-stage identity, rescues and blocks | Underlying frozen inputs, labels and settings | `results/paper/prose_checks.json`, `completeness_by_position.csv`, `overlap_with_independence_reference.csv` |

The experimental examples can be located by `prompt_row` and ordered mutator pair
in the frozen prompt export and corresponding response shards. No illustrative
attack example needs to be copied into this guide.

The map was checked against the following manuscript snapshots on 5 October 2026:

| File | SHA-256 |
| --- | --- |
| `TIFS_main_new.tex` | `4e14b262baddc8b2199ceaee62eafd2e18fdc29dca806cd2673965a251f865e5` |
| `TIFS_supplemental.tex` | `ebb5798c5f46d3795b3c178ee39ab0325e58fa25fc54cfaf5b42d76d926ddb8b` |

Manuscript edits require a new claim-to-artifact check. These hashes do not imply
that every interpretive statement has been mechanically proved.

## Data and join keys

`release/tifs-20261005/` is the unchanged 92-file numerical reference. Its
`manifest.json` pins every file. `release/paper-support/` is a separately
manifested extension recovered from the same completed sources. Neither should
be changed to make a failed verification pass.

| Support data | Contents and keys |
| --- | --- |
| `responses/{discovery,stability}/{run}/` | Target responses, keyed by `prompt_row` and `mutator`, or by `prompt_row`, `mutator_1`, `mutator_2`. Chain files are sharded by first mutator. All keys and response hashes match the original release. |
| `completeness/batch_{A,B,C}.csv.gz` | Original request, intermediate/final mutations, and both persistence labels. `source_row` is a within-file archival row index, not an asserted cross-batch AdvBench ID. Batch D uses the released frozen inputs and discovery persistence labels. |
| `writer_pilot/candidate_chains.csv.gz` | All 2,640 candidate chains, labels and provider-block statuses. GPT-3.5 reference generations are in the frozen chain export. |
| `writer_pilot/settings.json` | Recorded twenty-prompt selection, seed, models, prompt hashes and original descriptive comparison thresholds. |
| `validation/persistence_check_gt.csv` | The existing 132 human-annotated chains, expanded into 264 position-specific decisions. |
| `validation/intent_source_metadata.csv` | All 1,361 StrongREJECT row IDs, human median scores, source-model labels, and 39 opaque prompt clusters. No external prompt/response corpus is republished. |
| `provenance.json` | Source hashes and extraction scope. Local paths, endpoints and credentials are excluded. |

The extractor joins stability sources to the released allowlist before exporting
responses. Thus the support dataset contains only the seven reported conditions
and their required component baselines, even when a private source database had
additional work. `build_support_data.py` is a maintainer extraction tool, not a
dependency for user reproduction. Its optional archive and source-map inputs
are not needed once the released support data are present.

One frozen intermediate prompt is empty. Its final target input is nonempty.
This row is preserved rather than regenerated or silently removed. Provider
blocks likewise stay in the all-520 denominator as failed attacks.

## Analysis entry points

```bash
# Whole included evidence set, including saved response-hash checks and figures
python scripts/reproduce_paper.py --output-dir outputs/paper

# Optional whole-paper coverage check; broader than the distributed artifact
python scripts/reproduce_paper.py --no-figures --require-complete

# Full three-judge statistics and original clustered intervals
python scripts/analyze_intent_judge_validation.py
python scripts/analyze_persistence_judge_validation.py

# Original writer-pilot point estimates and bootstrap intervals
python scripts/analyze_mutator_alignment_pilot.py

# Figures alone
python scripts/plot_results.py --output-dir outputs/figures
```

The intent analysis resamples the 39 original prompt clusters, not individual
responses. Persistence resamples 132 chains, preserving the two decisions in
each chain. Both use 10,000 replicates and seed 20260914. The writer analysis
resamples the twenty selected prompt identifiers with 2,000 replicates and seed
20260902. All CSV metric outputs and the writer summary are compared to their
canonical references at floating-point tolerance `1e-12`.

To analyze new validation predictions, first execute the three judges in
separate checkpoint directories. Collect their `predictions_*.csv` files into
one analysis input directory, then run:

```bash
python scripts/analyze_intent_judge_validation.py \
  --input-dir outputs/intent-rerun --output-dir outputs/intent-rerun/analysis
python scripts/analyze_persistence_judge_validation.py \
  --input-dir outputs/persistence-rerun --output-dir outputs/persistence-rerun/analysis
python scripts/analyze_mutator_alignment_pilot.py \
  --run-dir outputs/writer-rerun --output-dir outputs/writer-rerun/analysis
```

The validation runners keep one checkpoint per judge. Their run-identity guards
prevent changing the judge
or dataset inside an existing checkpoint directory.

Fresh target runs are analyzed with `analyze_target_run.py`. This command checks
completion and equal prompt sets and labels its outputs noncanonical. New
judgments or target responses must not be mixed into the paper's frozen data.

## What is and is not reproduced

The original empirical results are recomputed from saved observations. Fresh
target evaluation reuses the frozen mutations and their persistence labels,
matching the paper's cross-target design. Regenerating those historical prompts
would be a different experiment. Some earlier batch configurations were unpinned
and changed during collection; their complete original generation environment
cannot be reconstructed. The released rows suffice to recompute their reported
descriptive completeness statistics.

The response archive now permits independent safety/intent rejudging. This is
separate from validating the judges' correctness. The persistence labels are
development annotations drawn from one original intent, not a held-out human
study. StrongREJECT provides an external human-effectiveness proxy, not new
annotations of these four target models. Its exact text corpus is fetched and
hash-checked only when fresh inference is explicitly requested.

The seven-condition stability panel was selected retrospectively and supports
only condition-specific descriptive conclusions. Raw ASR is independent of
persistence screening. Completeness belongs to the mutation batch, not to the
target model. The writer comparison cannot establish target-ASR effects for an
alternative writer because it did not evaluate those outputs on the targets.

The coverage report distinguishes successful reproduction of included evidence
from complete coverage of every manuscript claim. Independent benchmark
validation of the safety judge is outside the distributed artifact. Accordingly,
`--require-complete` returns a nonzero status even when all included analyses
pass. The manuscript is maintained separately and is not modified by repository
documentation updates.

The support dataset's provenance wording was revised for this repository scope.
Its manifest records the metadata revision; all experimental observations and
source-file hashes are unchanged. The original 92-file release remains unchanged.
