#!/usr/bin/env python3
"""Analyze persistence judges against the existing 264 human decisions."""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from release_inputs import DEFAULT_RELEASE
from paper_support import output_directory
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
)


DEFAULT_OUTPUT = Path("outputs/validation/persistence")
JUDGES = ["gpt4o_mini", "qwen", "gemma"]
DISPLAY = {
    "gpt4o_mini": "GPT-4o Mini",
    "qwen": "Qwen3.8 27B",
    "gemma": "Gemma 4 31B",
    "majority": "Three-judge majority",
}


def as_bool(series: pd.Series) -> pd.Series:
    values = series.astype(str).str.upper()
    if not values.isin(["TRUE", "FALSE"]).all():
        raise RuntimeError(f"Invalid Boolean values: {sorted(values.unique())}")
    return values.eq("TRUE")


def metrics(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float | int]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[False, True]).ravel()
    n = tn + fp + fn + tp
    if (tn + fp) and (tp + fn):
        balanced_accuracy = 0.5 * (tn / (tn + fp) + tp / (tp + fn))
    else:
        balanced_accuracy = float("nan")
    observed = (tn + tp) / n
    expected = ((tn + fp) * (tn + fn) + (fn + tp) * (fp + tp)) / (n * n)
    cohen_kappa = (
        (observed - expected) / (1 - expected)
        if expected != 1
        else float("nan")
    )
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
    mcc_denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = (tp * tn - fp * fn) / mcc_denominator if mcc_denominator else 0.0
    return {
        "n": len(y_true),
        "human_positive": int(y_true.sum()),
        "predicted_positive": int(y_pred.sum()),
        "accuracy": observed,
        "balanced_accuracy": balanced_accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mcc": mcc,
        "cohen_kappa": cohen_kappa,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def cluster_bootstrap(
    frame: pd.DataFrame,
    prediction: str,
    replicates: int,
) -> dict[str, tuple[float, float]]:
    rng = np.random.default_rng(20260914)
    cluster_counts = []
    for _, group in frame.groupby("chain_id", sort=False):
        cluster_counts.append(
            confusion_matrix(
                group["human_label"], group[prediction], labels=[False, True]
            ).ravel()
        )
    counts = np.asarray(cluster_counts, dtype=float)
    sampled = rng.integers(0, len(counts), size=(replicates, len(counts)))
    tn, fp, fn, tp = counts[sampled].sum(axis=1).T
    n = tn + fp + fn + tp
    accuracy = (tn + tp) / n
    specificity = np.divide(tn, tn + fp, out=np.zeros_like(tn), where=(tn + fp) > 0)
    sensitivity = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) > 0)
    balanced = 0.5 * (specificity + sensitivity)
    f1 = np.divide(2 * tp, 2 * tp + fp + fn, out=np.zeros_like(tp), where=(2 * tp + fp + fn) > 0)
    observed = accuracy
    expected = ((tn + fp) * (tn + fn) + (fn + tp) * (fp + tp)) / (n * n)
    kappa = np.divide(observed - expected, 1 - expected, out=np.zeros_like(observed), where=(1 - expected) != 0)
    return {
        key: (float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975)))
        for key, values in {
            "accuracy": accuracy,
            "balanced_accuracy": balanced,
            "f1": f1,
            "cohen_kappa": kappa,
        }.items()
    }


def format_pct(value: float) -> str:
    return "NA" if np.isnan(value) else f"{100 * value:.2f}%"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_RELEASE / "validation/persistence")
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    args = parser.parse_args()
    args.output_dir = output_directory(args.output_dir)

    frame: pd.DataFrame | None = None
    for judge in JUDGES:
        path = args.input_dir / f"predictions_{judge}.csv"
        current = pd.read_csv(path)
        if len(current) != 264 or current["prediction"].isna().any():
            raise RuntimeError(f"Expected 264 complete predictions in {path}")
        current["human_label"] = as_bool(current["human_label"])
        current[judge] = as_bool(current["prediction"])
        metadata = current[[
            "decision_id", "chain_id", "position", "mutator", "human_label"
        ]]
        if frame is None:
            frame = metadata.assign(**{judge: current[judge]})
        else:
            if not frame[metadata.columns].equals(metadata):
                raise RuntimeError(f"Metadata mismatch in {path}")
            frame[judge] = current[judge]
    assert frame is not None
    frame["majority"] = frame[JUDGES].sum(axis=1).ge(2)

    overall_rows = []
    for judge in JUDGES + ["majority"]:
        row = {"judge": judge, **metrics(frame["human_label"], frame[judge])}
        intervals = cluster_bootstrap(frame, judge, args.bootstrap_replicates)
        for metric, (low, high) in intervals.items():
            row[f"{metric}_ci_low"] = low
            row[f"{metric}_ci_high"] = high
        overall_rows.append(row)
    overall = pd.DataFrame(overall_rows)
    overall.to_csv(args.output_dir / "overall_metrics.csv", index=False)

    position_rows = []
    for position, group in frame.groupby("position", sort=False):
        for judge in JUDGES + ["majority"]:
            position_rows.append(
                {"position": position, "judge": judge, **metrics(group["human_label"], group[judge])}
            )
    by_position = pd.DataFrame(position_rows)
    by_position.to_csv(args.output_dir / "metrics_by_position.csv", index=False)

    mutator_rows = []
    for mutator, group in frame.groupby("mutator", sort=True):
        for judge in JUDGES + ["majority"]:
            mutator_rows.append(
                {"mutator": mutator, "judge": judge, **metrics(group["human_label"], group[judge])}
            )
    by_mutator = pd.DataFrame(mutator_rows)
    by_mutator.to_csv(args.output_dir / "metrics_by_mutator.csv", index=False)

    cell_rows = []
    for (mutator, position), group in frame.groupby(["mutator", "position"], sort=True):
        for judge in JUDGES + ["majority"]:
            cell_rows.append(
                {
                    "mutator": mutator,
                    "position": position,
                    "judge": judge,
                    **metrics(group["human_label"], group[judge]),
                }
            )
    pd.DataFrame(cell_rows).to_csv(
        args.output_dir / "metrics_by_mutator_and_position.csv", index=False
    )

    agreement_rows = []
    for first, second in combinations(JUDGES, 2):
        agreement_rows.append(
            {
                "judge_1": first,
                "judge_2": second,
                "agreement": accuracy_score(frame[first], frame[second]),
                "cohen_kappa": cohen_kappa_score(frame[first], frame[second]),
            }
        )
    agreement = pd.DataFrame(agreement_rows)
    agreement.to_csv(args.output_dir / "pairwise_agreement.csv", index=False)

    error_columns = ["decision_id", "chain_id", "position", "mutator", "human_label", *JUDGES, "majority"]
    frame.loc[frame["majority"].ne(frame["human_label"]), error_columns].to_csv(
        args.output_dir / "majority_errors.csv", index=False
    )
    frame.loc[frame[JUDGES].nunique(axis=1).gt(1), error_columns].to_csv(
        args.output_dir / "judge_disagreements.csv", index=False
    )

    lines = [
        "# Persistence-Classifier Validation",
        "",
        "The existing human annotation contains 132 ordered-pair chains. Each chain contributes one M1 label and one M2 label, giving 264 decisions. M1 compares the original prompt with the final chained prompt. M2 compares the intermediate prompt with the final chained prompt.",
        "",
        "The labels are imbalanced. M1 contains 39 positive and 93 negative labels, while M2 contains 125 positive and 7 negative labels. Balanced accuracy and the confusion counts are therefore essential alongside raw accuracy.",
        "",
        "## Overall Results",
        "",
        "| Judge | Accuracy | Balanced accuracy | Precision | Recall | F1 | Cohen kappa | TN / FP / FN / TP |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, row in overall.iterrows():
        lines.append(
            f"| {DISPLAY[row['judge']]} | {format_pct(row['accuracy'])} "
            f"[{format_pct(row['accuracy_ci_low'])}, {format_pct(row['accuracy_ci_high'])}] | "
            f"{format_pct(row['balanced_accuracy'])} | {format_pct(row['precision'])} | "
            f"{format_pct(row['recall'])} | {format_pct(row['f1'])} | "
            f"{row['cohen_kappa']:.3f} | {int(row['tn'])} / {int(row['fp'])} / "
            f"{int(row['fn'])} / {int(row['tp'])} |"
        )

    lines.extend([
        "",
        "## Results by Chain Position",
        "",
        "| Position | Judge | Human positive | Accuracy | Balanced accuracy | TN / FP / FN / TP |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ])
    for _, row in by_position.iterrows():
        lines.append(
            f"| {row['position']} | {DISPLAY[row['judge']]} | "
            f"{int(row['human_positive'])}/{int(row['n'])} | {format_pct(row['accuracy'])} | "
            f"{format_pct(row['balanced_accuracy'])} | {int(row['tn'])} / {int(row['fp'])} / "
            f"{int(row['fn'])} / {int(row['tp'])} |"
        )

    lines.extend([
        "",
        "## Results by Mutator",
        "",
        "Each mutator has 22 labels. These accuracies are diagnostic because the per-mutator samples are small.",
        "",
        "| Mutator | Human positive | GPT-4o Mini | Qwen3.8 27B | Gemma 4 31B | Majority |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ])
    for mutator in sorted(frame["mutator"].unique()):
        subset = by_mutator.loc[by_mutator["mutator"].eq(mutator)].set_index("judge")
        lines.append(
            f"| {mutator} | {int(subset.loc['gpt4o_mini', 'human_positive'])}/22 | "
            f"{format_pct(subset.loc['gpt4o_mini', 'accuracy'])} | "
            f"{format_pct(subset.loc['qwen', 'accuracy'])} | "
            f"{format_pct(subset.loc['gemma', 'accuracy'])} | "
            f"{format_pct(subset.loc['majority', 'accuracy'])} |"
        )

    lines.extend([
        "",
        "## Pairwise Judge Agreement",
        "",
        "| Judge 1 | Judge 2 | Agreement | Cohen kappa |",
        "| --- | --- | ---: | ---: |",
    ])
    for _, row in agreement.iterrows():
        lines.append(
            f"| {DISPLAY[row['judge_1']]} | {DISPLAY[row['judge_2']]} | "
            f"{format_pct(row['agreement'])} | {row['cohen_kappa']:.3f} |"
        )

    lines.extend([
        "",
        "## Interpretation Boundary",
        "",
        "This is a retrospective validation on the study's existing annotations, not a newly sampled or independently held-out human study. Each mutator has 22 labels overall and only 11 labels in each position, so per-mutator estimates are diagnostic rather than precise. The source labels may also have informed development of the current evaluator prompts.",
        "",
        "The validation directly tests the persistence classifier used to form completeness-based screens. It does not alter raw all-520 target-model ASR, which does not depend on persistence labels.",
        "",
        "Machine-readable overall, position, mutator, mutator-position, agreement, disagreement, and error tables are stored beside this summary.",
        "",
    ])
    (args.output_dir / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")

    manifest = {
        "status": "PASS",
        "source_chains": int(frame["chain_id"].nunique()),
        "decisions": len(frame),
        "human_positive": int(frame["human_label"].sum()),
        "human_negative": int((~frame["human_label"]).sum()),
        "m1_positive": int(frame.loc[frame["position"].eq("M1"), "human_label"].sum()),
        "m2_positive": int(frame.loc[frame["position"].eq("M2"), "human_label"].sum()),
        "majority_errors": int(frame["majority"].ne(frame["human_label"]).sum()),
        "judge_disagreements": int(frame[JUDGES].nunique(axis=1).gt(1).sum()),
        "bootstrap_replicates": args.bootstrap_replicates,
        "bootstrap_cluster": "chain_id",
    }
    (args.output_dir / "analysis_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print((args.output_dir / "SUMMARY.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
