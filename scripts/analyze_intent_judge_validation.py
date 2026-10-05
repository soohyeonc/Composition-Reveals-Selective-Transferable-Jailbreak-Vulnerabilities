#!/usr/bin/env python3
"""Analyze three intent judges against StrongREJECT human-effectiveness labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from release_inputs import DEFAULT_RELEASE
from paper_support import SUPPORT, output_directory
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)


DEFAULT_OUTPUT = Path("outputs/validation/intent")
JUDGES = ["gpt4o_mini", "qwen", "gemma"]
DISPLAY = {
    "gpt4o_mini": "GPT-4o Mini",
    "qwen": "Qwen3.8 27B",
    "gemma": "Gemma 4 31B",
    "majority": "Three-judge majority",
}


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | int]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[False, True]).ravel()
    return {
        "n": len(y_true),
        "human_positive": int(y_true.sum()),
        "predicted_positive": int(y_pred.sum()),
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "cohen_kappa": cohen_kappa_score(y_true, y_pred),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def cluster_bootstrap(
    frame: pd.DataFrame,
    truth_column: str,
    prediction_column: str,
    replicates: int = 10000,
) -> dict[str, tuple[float, float]]:
    rng = np.random.default_rng(20260914)
    cluster_counts = []
    for _, group in frame.groupby("forbidden_prompt", sort=False):
        tn, fp, fn, tp = confusion_matrix(
            group[truth_column], group[prediction_column], labels=[False, True]
        ).ravel()
        cluster_counts.append([tn, fp, fn, tp])
    counts = np.asarray(cluster_counts, dtype=float)
    sampled = rng.integers(0, len(counts), size=(replicates, len(counts)))
    totals = counts[sampled].sum(axis=1)
    tn, fp, fn, tp = totals.T
    n = totals.sum(axis=1)
    accuracy = (tn + tp) / n
    balanced_accuracy = 0.5 * (
        np.divide(tn, tn + fp, out=np.zeros_like(tn), where=(tn + fp) > 0)
        + np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) > 0)
    )
    f1 = np.divide(2 * tp, 2 * tp + fp + fn, out=np.zeros_like(tp), where=(2 * tp + fp + fn) > 0)
    observed = accuracy
    expected = ((tn + fp) * (tn + fn) + (fn + tp) * (fp + tp)) / (n * n)
    kappa = np.divide(
        observed - expected,
        1 - expected,
        out=np.zeros_like(observed),
        where=(1 - expected) != 0,
    )
    values = {
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "f1": f1,
        "cohen_kappa": kappa,
    }
    return {
        key: (float(np.quantile(value, 0.025)), float(np.quantile(value, 0.975)))
        for key, value in values.items()
    }


def format_pct(value: float) -> str:
    return f"{100 * value:.2f}%"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_RELEASE / "validation/intent")
    parser.add_argument("--source-metadata", type=Path, default=SUPPORT / "validation/intent_source_metadata.csv")
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    args = parser.parse_args()

    args.output_dir = output_directory(args.output_dir)
    source = pd.read_csv(args.source_metadata)
    frame = source.rename(columns={"cluster_id": "forbidden_prompt"})
    if len(frame) != 1361 or frame.row_id.duplicated().any():
        raise ValueError("Expected all 1,361 unique validation rows")
    for judge in JUDGES:
        prediction_path = args.input_dir / f"predictions_{judge}.csv"
        predictions = pd.read_csv(prediction_path, usecols=["row_id", "prediction"])
        if len(predictions) != len(frame) or predictions["prediction"].isna().any():
            raise RuntimeError(f"Missing predictions in {prediction_path}")
        frame = frame.merge(
            predictions.rename(columns={"prediction": judge}), on="row_id", how="left", validate="one_to_one"
        )
        if not frame[judge].astype(str).str.upper().isin(["TRUE", "FALSE"]).all():
            raise ValueError("Missing or invalid judge labels")
        # read_csv may infer the literal TRUE/FALSE column as bool. Normalize
        # through text so both bool and string representations are handled.
        frame[judge] = frame[judge].astype(str).str.upper().eq("TRUE")

    frame["human_nonzero"] = frame["label_median"].gt(0)
    frame["human_half"] = frame["label_median"].ge(0.5)
    frame["human_full"] = frame["label_median"].eq(1)
    frame["majority"] = frame[JUDGES].sum(axis=1).ge(2)

    primary_rows = []
    intervals = {}
    for judge in JUDGES + ["majority"]:
        result = {"judge": judge, **metrics(frame["human_nonzero"].to_numpy(), frame[judge].to_numpy())}
        ci = cluster_bootstrap(
            frame,
            "human_nonzero",
            judge,
            replicates=args.bootstrap_replicates,
        )
        intervals[judge] = ci
        for metric, (low, high) in ci.items():
            result[f"{metric}_ci_low"] = low
            result[f"{metric}_ci_high"] = high
        primary_rows.append(result)
    primary = pd.DataFrame(primary_rows)
    primary.to_csv(args.output_dir / "primary_metrics.csv", index=False)

    threshold_rows = []
    for truth in ["human_nonzero", "human_half", "human_full"]:
        for judge in JUDGES + ["majority"]:
            threshold_rows.append(
                {"human_threshold": truth, "judge": judge, **metrics(frame[truth].to_numpy(), frame[judge].to_numpy())}
            )
    pd.DataFrame(threshold_rows).to_csv(
        args.output_dir / "threshold_sensitivity.csv", index=False
    )

    agreement_rows = []
    for i, first in enumerate(JUDGES):
        for second in JUDGES[i + 1 :]:
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

    by_model_rows = []
    for model, group in frame.groupby("model"):
        for judge in JUDGES:
            by_model_rows.append(
                {"source_model": model, "judge": judge, **metrics(group["human_nonzero"].to_numpy(), group[judge].to_numpy())}
            )
    pd.DataFrame(by_model_rows).to_csv(args.output_dir / "metrics_by_source_model.csv", index=False)

    score_rows = []
    for score, group in frame.groupby("label_median"):
        row = {"human_median": score, "n": len(group)}
        for judge in JUDGES:
            row[f"{judge}_true_rate"] = group[judge].mean()
        score_rows.append(row)
    pd.DataFrame(score_rows).to_csv(args.output_dir / "true_rate_by_human_score.csv", index=False)

    disagreements = frame.loc[
        frame[JUDGES].nunique(axis=1).gt(1),
        ["row_id", "model", "jailbreak", "label_median", *JUDGES],
    ]
    disagreements.to_csv(args.output_dir / "judge_disagreements.csv", index=False)

    lines = [
        "# Intent-Judge Validation on StrongREJECT Human Evaluations",
        "",
        "The primary reference is whether the median human jailbreak-effectiveness score is nonzero. This is a compatible proxy for attempted intent fulfillment, not an exact human annotation of the paper's binary intent definition.",
        "",
        f"The dataset contains {len(frame):,} prompt-response pairs, {frame['forbidden_prompt'].nunique()} unique forbidden prompts, and up to five human ratings per pair.",
        "",
        "## Primary Results",
        "",
        "| Judge | Accuracy | Balanced accuracy | Precision | Recall | F1 | Cohen kappa | Confusion (TN/FP/FN/TP) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, row in primary.iterrows():
        lines.append(
            f"| {DISPLAY[row['judge']]} | {format_pct(row['accuracy'])} "
            f"[{format_pct(row['accuracy_ci_low'])}, {format_pct(row['accuracy_ci_high'])}] | "
            f"{format_pct(row['balanced_accuracy'])} | {format_pct(row['precision'])} | "
            f"{format_pct(row['recall'])} | {format_pct(row['f1'])} | "
            f"{row['cohen_kappa']:.3f} | {int(row['tn'])}/{int(row['fp'])}/{int(row['fn'])}/{int(row['tp'])} |"
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
        "StrongREJECT humans rated jailbreak effectiveness on a five-point scale. The study's intent judge instead asks whether a response attempts to fulfill the original request even when incorrect or incomplete. Agreement therefore measures external convergent validity. It must not be described as exact intent-classification accuracy.",
        "",
        "Threshold sensitivity, source-model breakdowns, judge disagreement rows, and machine-readable confusion counts are included beside this summary.",
        "",
    ])
    (args.output_dir / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")

    manifest = {
        "status": "PASS",
        "rows": len(frame),
        "unique_prompts": int(frame["forbidden_prompt"].nunique()),
        "primary_human_reference": "label_median > 0",
        "primary_human_positive": int(frame["human_nonzero"].sum()),
        "primary_human_negative": int((~frame["human_nonzero"]).sum()),
        "judge_disagreements": len(disagreements),
        "bootstrap_replicates": args.bootstrap_replicates,
    }
    (args.output_dir / "analysis_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print((args.output_dir / "SUMMARY.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
