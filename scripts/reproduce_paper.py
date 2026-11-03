#!/usr/bin/env python3
"""Recompute paper tables, validation intervals, figures, and evidence coverage.

Offline only. Missing historical evidence is reported, never silently inferred.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from release_inputs import DEFAULT_RELEASE, MODELS, ROOT, check_manifest
from paper_support import SUPPORT, check_support, output_directory
from verify_tifs_release import exclude_refusals, joined, load, refusal_flags, refused, reproduce, stability_markdown, truth


def compare_csv(actual, reference):
    pd.testing.assert_frame_equal(pd.read_csv(actual), pd.read_csv(reference),
                                  check_dtype=False, check_exact=False,
                                  atol=1e-12, rtol=1e-12)


def same_json(actual, reference):
    if isinstance(reference, dict):
        assert set(actual) == set(reference)
        for key in reference:
            same_json(actual[key], reference[key])
    elif isinstance(reference, (int, float)):
        assert np.isclose(actual, reference, atol=1e-12, rtol=1e-12), (actual, reference)
    else:
        assert actual == reference, (actual, reference)


def verify_responses(release, support):
    rows = []
    for phase in ["discovery", "stability"]:
        for directory in sorted((support / "responses" / phase).iterdir()):
            for table in ["baselines", "chains"]:
                keys = ["prompt_row"] + (["mutator"] if table == "baselines" else ["mutator_1", "mutator_2"])
                paths = list(directory.glob(f"{table}*.csv.gz"))
                actual = pd.concat([pd.read_csv(p, keep_default_na=False) for p in paths], ignore_index=True)
                expected = pd.read_csv(release / phase / f"{directory.name}_{table}.csv.gz", keep_default_na=False)
                assert len(actual) == len(expected) and not actual.duplicated(keys).any()
                merged = expected[keys + ["target_response_sha256"]].merge(actual, on=keys, validate="one_to_one")
                assert len(merged) == len(expected)
                hashes = merged.target_response.map(lambda x: hashlib.sha256(x.encode()).hexdigest())
                assert hashes.eq(merged.target_response_sha256).all(), directory.name
                rows.append({"phase": phase, "run": directory.name, "kind": table, "hash_matched_responses": len(merged)})
    return pd.DataFrame(rows)


def completeness_tables(release, support, out):
    batches, rates = [], []
    for batch in "ABCD":
        if batch == "D":
            frame = pd.read_csv(release / "discovery/luna_chains.csv.gz")
        else:
            frame = pd.read_csv(support / f"completeness/batch_{batch}.csv.gz", keep_default_na=False)
        frame["complete"] = truth(frame.m1_persistence) & truth(frame.m2_persistence)
        grouped = frame.groupby(["mutator_1", "mutator_2"]).complete.agg(["size", "sum", "mean"])
        assert len(grouped) == 132
        rates.append(grouped["mean"].rename(f"batch_{batch.lower()}_rate"))
        median = grouped["sum"].median()
        batches.append({"batch": batch, "evaluated_rows": len(frame),
                        "median_complete_count": median,
                        "retained_pairs": int(grouped["sum"].ge(median).sum())})
    batch_table = pd.DataFrame(batches)
    reference_summary = json.loads((release / "completeness/summary.json").read_text())
    for actual, expected in zip(batches, reference_summary["batches"]):
        for key in actual:
            assert actual[key] == expected[key], (key, actual, expected)
    matrix = pd.concat(rates, axis=1)
    expected = pd.read_csv(release / "completeness/pair_metrics.csv").set_index(["m1", "m2"])
    assert np.allclose(matrix, expected.loc[matrix.index, matrix.columns])
    matrix["mean_complete_count"] = matrix.mean(axis=1) * 520
    matrix["sd_complete_count"] = matrix.iloc[:, :4].std(axis=1, ddof=1) * 520
    matrix["retained_at_mean_median"] = matrix.mean_complete_count.ge(matrix.mean_complete_count.median())
    assert np.allclose(matrix.mean_complete_count, expected.loc[matrix.index, "mean_complete_count"])
    assert np.allclose(matrix.sd_complete_count, expected.loc[matrix.index, "sd_complete_count"])
    batch_table.to_csv(out / "table05_completeness_batches.csv", index=False)
    matrix.reset_index().to_csv(out / "completeness_pair_metrics.csv", index=False)
    marginal = []
    for position, level in [("M1", 0), ("M2", 1)]:
        for mutator, group in matrix.groupby(level=level):
            marginal.append({"mutator": mutator, "position": position,
                             "mean_complete_count": group.mean_complete_count.mean(),
                             "retained_pairs": int(group.retained_at_mean_median.sum()), "pairs": len(group)})
    pd.DataFrame(marginal).to_csv(out / "completeness_by_position.csv", index=False)
    return {
        "grand_mean_complete_count": float(matrix.mean_complete_count.mean()),
        "median_mean_complete_count": float(matrix.mean_complete_count.median()),
        "mean_cell_sd_complete_count": float(matrix.sd_complete_count.mean()),
        "retained_pairs": int(matrix.retained_at_mean_median.sum()),
        "pairwise_correlations": matrix.iloc[:, :4].corr().to_dict(),
    }


def write_tables(release, support, out, tables):
    panel = json.loads((release / "panel.json").read_text())["models"]
    pairs = tables["pair_metrics.csv"].copy()
    runs, conditions = tables["stability_runs.csv"], tables["stability_summary.csv"]
    flags = refusal_flags(release)
    models = tables["model_summary.csv"].copy()
    models["individual_asr_percent"] = 100 * models.individual_successes / models.individual_attempts
    models["raw_pair_asr_percent"] = 100 * models.pair_successes / models.pair_attempts
    models["raw_positive_pair_percent"] = 100 * models.raw_positive_pairs / 132
    models["matched_positive_pair_percent"] = 100 * models.matched_screen_pairs / 132
    models.to_csv(out / "table06_model_summary.csv", index=False)
    pairs["raw_pair_asr_percent"] = 100 * pairs.pair_successes / pairs.prompts
    pairs["stronger_baseline_successes"] = pairs[["m1_successes", "m2_successes"]].max(axis=1)
    pairs["raw_multiplier"] = pairs.pair_successes / pairs.stronger_baseline_successes.replace(0, np.nan)
    pairs.to_csv(out / "all_discovery_pairs.csv", index=False)
    featured, confirmation, per_run, baselines = [], [], [], []
    assertions = {}
    for slug, model in MODELS.items():
        positives = conditions.loc[conditions.model.eq(model) & conditions.role.eq("positive")]
        feature = positives.sort_values("multiplier", ascending=False).iloc[0]
        a, b = feature.mutator_1, feature.mutator_2
        discovery = pairs.loc[pairs.model.eq(model) & pairs.mutator_1.eq(a) & pairs.mutator_2.eq(b)].iloc[0]
        featured.append({"model": model, "mutator_1": a, "mutator_2": b, "attempts": int(discovery.prompts),
                         "pair_successes": int(discovery.pair_successes),
                         "stronger_baseline_successes": int(discovery.stronger_baseline_successes),
                         "multiplier": discovery.raw_multiplier,
                         "pair_only_rescues": int(discovery.pair_only_rescues)})
        confirmation.append({**feature.to_dict(), "positives_retained": int(positives.outcome.eq("Retained").sum()), "positives_reported": 5})
        selected = runs.loc[runs.model.eq(model) & runs.mutator_1.eq(a) & runs.mutator_2.eq(b)].copy()
        selected["stronger_component_successes"] = selected[["m1_successes", "m2_successes"]].max(axis=1)
        assert selected.m2_successes.ge(selected.m1_successes).all()
        per_run.append(selected)
        base, _ = exclude_refusals(load(release / f"discovery/{slug}_baselines.csv.gz", 12, True),
                                   load(release / f"discovery/{slug}_chains.csv.gz", 132), flags)
        component = base.loc[~base.refused].groupby("mutator").success.agg(["size", "sum"]).reset_index()
        component.columns = ["mutator", "attempts", "successes"]
        component.insert(0, "model", model)
        component["asr_percent"] = 100 * component.successes / component.attempts
        baselines.append(component)
        own_runs = runs.loc[runs.model.eq(model)].merge(conditions.loc[conditions.model.eq(model), ["mutator_1", "mutator_2", "original"]], on=["mutator_1", "mutator_2"], validate="many_to_one")
        attempts, blocks = 0, 0
        for run in range(1, 6):
            b = load(release / f"stability/{slug}_run{run}_baselines.csv.gz", len(set(runs.loc[runs.model.eq(model), "mutator_1"]) | set(runs.loc[runs.model.eq(model), "mutator_2"])), True)
            b, c = exclude_refusals(b, joined(b, load(release / f"stability/{slug}_run{run}_chains.csv.gz", 7)), flags)
            for frame in [b.loc[~b.refused], c.loc[~c.refused]]:
                attempts += len(frame)
                blocks += int(frame.blocked.sum())
        assertions[slug] = {"max_discovery_rerun_count_difference": int((own_runs.pair_successes - own_runs.original).abs().max()),
                            "stability_target_tasks": attempts, "stability_provider_blocks": blocks}
        assert assertions[slug]["max_discovery_rerun_count_difference"] <= 19
    pd.DataFrame(featured).to_csv(out / "table07_featured_discovery.csv", index=False)
    pd.DataFrame(confirmation).to_csv(out / "table08_featured_stability.csv", index=False)
    pd.concat(per_run).to_csv(out / "table09_featured_runs.csv", index=False)
    pd.concat(baselines).to_csv(out / "individual_mutator_results.csv", index=False)
    conditions.to_csv(out / "table11_all_stability_conditions.csv", index=False)
    (out / "STABILITY_TABLES.md").write_text(stability_markdown(release, runs, tables["pair_metrics.csv"]))
    return assertions


def counts(series):
    return {key: int(value) for key, value in series.items()}


def writer_refusals(release, support):
    keys = ["prompt_row", "mutator_1", "mutator_2"]
    labels = pd.read_csv(release / "discovery/luna_chains.csv.gz")
    frozen = pd.read_csv(release / "prompts/frozen_chains.csv.gz", keep_default_na=False).merge(
        labels[keys + ["m1_persistence", "m2_persistence"]], on=keys, validate="one_to_one")
    first, second = refused(frozen.jailbreak_prompt_1), refused(frozen.jailbreak_prompt_2)
    either = first | second
    base = pd.read_csv(release / "prompts/frozen_baselines.csv.gz", keep_default_na=False)
    standalone = refused(base.mutated_prompt)
    candidate = pd.read_csv(support / "writer_pilot/candidate_chains.csv.gz", keep_default_na=False)
    candidate["blocked"] = candidate.provider_status.ne("ok")
    candidate["refused_first"] = ~candidate.blocked & refused(candidate.jailbreak_prompt_1)
    pilot = pd.read_csv(release / "writer_pilot/row_labels.csv").merge(
        candidate[keys + ["blocked", "refused_first"]], on=keys, validate="one_to_one")
    kept = pilot.loc[~pilot.blocked & ~pilot.refused_first]
    on_pilot_rows = frozen.prompt_row.isin(set(pilot.prompt_row))
    result = {
        "frozen_chains": {"chains": len(frozen), "first_stage": int(first.sum()), "second_stage": int(second.sum()),
                          "either": int(either.sum()),
                          "complete": int((either & truth(frozen.m1_persistence) & truth(frozen.m2_persistence)).sum())},
        "frozen_standalone": {"prompts": len(base), "refused": int(standalone.sum()),
                              "by_mutator": counts(base.loc[standalone].mutator.value_counts())},
        "candidate_provider_blocks": int(candidate.blocked.sum()),
        "candidate_first_stage_refusals": int(candidate.refused_first.sum()),
        "candidate_first_stage_refusals_by_mutator": counts(candidate.groupby("mutator_1").refused_first.sum()),
        "reference_refusals_on_pilot_chains": int((either & on_pilot_rows).sum()),
        "pilot_chains": int(on_pilot_rows.sum()),
        "rows_without_block_or_first_stage_refusal": len(kept),
        "candidate_complete_on_those_rows": int((truth(kept.m1_persistence_candidate) & truth(kept.m2_persistence_candidate)).sum()),
        "reference_complete_on_those_rows": int((truth(kept.m1_persistence_reference) & truth(kept.m2_persistence_reference)).sum()),
    }
    assert (result["frozen_chains"]["either"], result["frozen_chains"]["complete"], result["frozen_standalone"]["refused"]) == (393, 0, 28)
    assert (result["candidate_provider_blocks"], result["candidate_first_stage_refusals"], len(kept)) == (109, 1698, 833)
    assert (result["candidate_complete_on_those_rows"], result["reference_complete_on_those_rows"]) == (193, 241)
    assert (result["reference_refusals_on_pilot_chains"], result["pilot_chains"]) == (1, 2640)
    return result


def obfuscation_first(release, support):
    batches = {}
    for batch in "ABCD":
        if batch == "D":
            frame = pd.read_csv(release / "discovery/luna_chains.csv.gz")
        else:
            frame = pd.read_csv(support / f"completeness/batch_{batch}.csv.gz", keep_default_na=False)
        frame = frame.loc[frame.mutator_1.eq("ea-obfuscation")]
        first, second = truth(frame.m1_persistence), truth(frame.m2_persistence)
        encryption = frame.mutator_2.eq("ea-encryption")
        batches[batch] = {"encryption_chains": int(encryption.sum()),
                          "encryption_first_retained": int((encryption & first).sum()),
                          "encryption_second_retained": int((encryption & second).sum()),
                          "encryption_first_only": int((encryption & first & ~second).sum()),
                          "encryption_second_only": int((encryption & ~first & second).sum()),
                          "other_chains": int((~encryption).sum()),
                          "other_first_retained": int((~encryption & first).sum())}
    assert min(b["encryption_first_retained"] for b in batches.values()) >= 507
    assert max(b["encryption_second_retained"] for b in batches.values()) <= 26
    d = batches["D"]
    assert (d["encryption_second_only"], d["encryption_first_only"], d["other_first_retained"], d["other_chains"]) == (4, 489, 27, 5200)
    return batches


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", type=Path, default=DEFAULT_RELEASE)
    parser.add_argument("--support", type=Path, default=SUPPORT)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/paper"))
    parser.add_argument("--no-figures", action="store_true")
    parser.add_argument("--require-complete", action="store_true", help="Exit 2 if any paper claim lacks source evidence")
    args = parser.parse_args()
    out = output_directory(args.output_dir)
    release, support = args.release.resolve(), args.support.resolve()
    release_hash, support_hash = check_manifest(release), check_support(support)
    tables = reproduce(release)
    for name, table in tables.items():
        table.to_csv(out / name, index=False)
        compare_csv(out / name, release / "derived" / name)
    print("PASS: discovery, seven-condition stability, and overlap tables", flush=True)
    responses = verify_responses(release, support)
    responses.to_csv(out / "response_audit.csv", index=False)
    print(f"PASS: {responses.hash_matched_responses.sum():,} hash-matched archived target responses", flush=True)
    claims = {"completeness": completeness_tables(release, support, out)}
    claims["stability"] = write_tables(release, support, out, tables)
    for kind in ["intent", "persistence"]:
        command = [sys.executable, str(ROOT / f"scripts/analyze_{kind}_judge_validation.py"),
                   "--input-dir", str(release / "validation" / kind), "--output-dir", str(out / "validation" / kind)]
        if kind == "intent":
            command += ["--source-metadata", str(support / "validation/intent_source_metadata.csv")]
        subprocess.run(command, check=True, capture_output=True, text=True)
        for expected in (release / "validation" / kind).glob("*.csv"):
            if not expected.name.startswith("predictions_"):
                compare_csv(out / "validation" / kind / expected.name, expected)
        filename = "primary_metrics.csv" if kind == "intent" else "overall_metrics.csv"
        shutil.copyfile(out / "validation" / kind / filename,
                        out / f"table0{'3' if kind == 'intent' else '4'}_{kind}_validation.csv")
        print(f"PASS: all {kind} judge metrics and 10,000-replicate cluster bootstrap intervals", flush=True)
    subprocess.run([sys.executable, str(ROOT / "scripts/analyze_mutator_alignment_pilot.py"),
                    "--release", str(release), "--support", str(support),
                    "--output-dir", str(out / "writer_pilot")], check=True, capture_output=True, text=True)
    for name in ["pair_metrics.csv", "per_mutator_persistence.csv"]:
        compare_csv(out / "writer_pilot" / name, release / "writer_pilot" / name)
    actual = json.loads((out / "writer_pilot/summary.json").read_text())
    same_json(actual, json.loads((release / "writer_pilot/summary.json").read_text()))
    pilot = pd.read_csv(release / "writer_pilot/row_labels.csv")
    writer_rows = []
    for writer in ["reference", "candidate"]:
        first, second = truth(pilot[f"m1_persistence_{writer}"]), truth(pilot[f"m2_persistence_{writer}"])
        writer_rows.append({"writer": "GPT-3.5" if writer == "reference" else "GPT-5.6 Luna",
                            "attempts": len(pilot), "m1_persistent": int(first.sum()),
                            "m2_persistent": int(second.sum()), "complete": int((first & second).sum()),
                            "complete_percent": 100 * (first & second).mean()})
    pd.DataFrame(writer_rows).to_csv(out / "table10_writer_pilot.csv", index=False)
    claims["writer_pilot"] = {**actual, "candidate_complete_upper_bound_with_all_blocks_successful":
        (writer_rows[1]["complete"] + actual["candidate_provider_blocks"]) / len(pilot)}
    frozen = pd.read_csv(release / "prompts/frozen_chains.csv.gz", keep_default_na=False)
    base = pd.read_csv(release / "prompts/frozen_baselines.csv.gz", keep_default_na=False)
    comparison = frozen.merge(base[["prompt_row", "mutator", "mutated_prompt"]],
                              left_on=["prompt_row", "mutator_1"], right_on=["prompt_row", "mutator"], validate="many_to_one")
    identical = int(comparison.jailbreak_prompt_1.eq(comparison.mutated_prompt).sum())
    assert identical == 294
    claims["frozen_first_stage_identity"] = {"identical": identical, "chains": len(frozen)}
    claims["writer_refusals"] = writer_refusals(release, support)
    pair_prompts = tables["pair_metrics.csv"].prompts
    summary = tables["model_summary.csv"]
    claims["refusal_exclusion"] = {
        "standalone_prompts_analyzed": int(summary.individual_attempts.iloc[0]),
        "chains_analyzed": int(summary.pair_attempts.iloc[0]),
        "pair_comparison_prompts_min": int(pair_prompts.min()), "pair_comparison_prompts_max": int(pair_prompts.max()),
        "stability_pair_prompts_sent": 72800, "stability_pair_prompts_analyzed": int(tables["stability_runs.csv"].attempts.sum())}
    assert claims["refusal_exclusion"] == {"standalone_prompts_analyzed": 6212, "chains_analyzed": 68247,
                                           "pair_comparison_prompts_min": 471, "pair_comparison_prompts_max": 520,
                                           "stability_pair_prompts_sent": 72800, "stability_pair_prompts_analyzed": 70935}
    claims["obfuscation_first_positions"] = obfuscation_first(release, support)
    claims["case_study"] = []
    for slug, model in MODELS.items():
        b = load(release / f"discovery/{slug}_baselines.csv.gz", 12, True)
        b, c = exclude_refusals(b, joined(b, load(release / f"discovery/{slug}_chains.csv.gz", 132)), refusal_flags(release))
        selected = c.loc[c.mutator_1.eq("mm-cognitive-hacking") & c.mutator_2.eq("mm-nshot-hacking") & ~c.refused]
        example = selected.loc[selected.prompt_row.eq(1)].iloc[0]
        claims["case_study"].append({"model": model, "prompt_row": 1, "pair_success": bool(example.success),
            "m1_success": bool(example.s1), "m2_success": bool(example.s2),
            "complete_pair_only_rescues": int((selected.success & ~selected.s1 & ~selected.s2 & selected.complete).sum())})
    overlap = tables["cross_target_overlap.csv"].copy()
    sizes = tables["model_summary.csv"].set_index("model").raw_positive_pairs
    overlap["independent_expected_overlap"] = [sizes[r.model_1] * sizes[r.model_2] / 132 for r in overlap.itertuples()]
    overlap["observed_expected_overlap_ratio"] = overlap.intersection / overlap.independent_expected_overlap
    # The paper reports Jaccard relative to the Jaccard expected from independent sets of the observed sizes.
    overlap["independent_expected_jaccard"] = [e / (sizes[r.model_1] + sizes[r.model_2] - e)
                                               for e, r in zip(overlap.independent_expected_overlap, overlap.itertuples())]
    overlap["observed_expected_jaccard_ratio"] = overlap.jaccard / overlap.independent_expected_jaccard
    overlap.to_csv(out / "overlap_with_independence_reference.csv", index=False)
    pd.json_normalize(json.loads((release / "run_settings.json").read_text())).to_csv(out / "table02_recorded_settings.csv", index=False)
    (out / "prose_checks.json").write_text(json.dumps(claims, indent=2) + "\n")
    if not args.no_figures:
        from plot_results import plot
        plot(release, out / "figures")
        shutil.copyfile(support / "figures/chained-mutator.png", out / "figures/chained-mutator.png")
    report = {
        "artifact_checks": "PASS", "paper_coverage": "INCOMPLETE",
        "release_manifest_sha256": release_hash, "support_manifest_sha256": support_hash,
        "response_records_checked": int(responses.hash_matched_responses.sum()),
        "reported_stability_conditions": 28, "reported_pair_runs": 140,
        "judge_bootstrap_replicates": 10000, "writer_bootstrap_replicates": 2000,
        "model_calls": 0,
        "missing_evidence": [{"scope": "Independent safety-judge benchmark validation",
                              "status": "Outside the distributed artifact"}],
        "external_input": {"kind": "StrongREJECT text for fresh judge inference only",
                           "url": "https://osf.io/download/jwmqe/",
                           "sha256": "7a5928a4f09b4cbfce274001873ccf095510d5a990f8be52e4323d4c069fbad2",
                           "offline_numeric_reproduction": "All metadata, clusters and labels included"},
    }
    (out / "coverage.json").write_text(json.dumps(report, indent=2) + "\n")
    print("PASS: all included evidence. Whole-paper coverage is broader than the distributed artifact; see coverage.json.")
    if args.require_complete:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
