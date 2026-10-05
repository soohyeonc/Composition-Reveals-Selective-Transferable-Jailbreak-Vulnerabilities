#!/usr/bin/env python3
"""Regenerate empirical paper panels from the immutable release; no model calls."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from release_inputs import DEFAULT_RELEASE, MODELS, check_manifest

LABELS = {
    "mm-gaslighting": "gas", "mm-cognitive-hacking": "ch",
    "mm-privilege-escalation": "pe", "ea-translation": "tr",
    "ea-fictional": "fic", "mm-forced-completion": "fc",
    "mm-prompt-injection": "pi", "mm-nshot-hacking": "nh",
    "mm-roleplay": "rp", "ea-paraphrasing": "pp",
    "ea-encryption": "enc", "ea-obfuscation": "obs",
}
SHORT_NAMES = dict(zip(MODELS.values(), ["Luna", "R1", "Gemma", "Qwen"]))


def matrix(frame, column, mask=None):
    order = frame.pivot(index="m1", columns="m2", values="complete_count")
    rows = order.mean(axis=1).sort_values(ascending=False).index
    columns = order.mean(axis=0).sort_values(ascending=False).index
    data = frame[["m1", "m2"]].copy()
    data["value"] = frame[column] if mask is None else frame[column].where(mask)
    return data.pivot(index="m1", columns="m2", values="value").loc[rows, columns].rename(
        index=LABELS, columns=LABELS)


def heatmap(data, output, maximum):
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(data, ax=ax, annot=True, fmt=".1f", cmap="viridis",
                vmin=0, vmax=maximum, cbar=False, linewidths=0.5, linecolor="white")
    ax.set_xlabel("Mutator 2", fontsize=28)
    ax.set_ylabel("Mutator 1", fontsize=28)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize=22)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=22)
    fig.tight_layout()
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot(release, output):
    release, output = Path(release).resolve(), Path(output).resolve()
    if output.is_relative_to(release) or release.is_relative_to(output):
        raise ValueError("Write generated figures outside the immutable release")
    check_manifest(release)
    output.mkdir(parents=True, exist_ok=True)
    completeness = pd.read_csv(release / "completeness/pair_metrics.csv").rename(
        columns={"mean_complete_count": "complete_count"})
    heatmap(matrix(completeness, "complete_count"), output / "fig3a_mean_completeness.png", 520)
    heatmap(matrix(completeness, "complete_count", completeness.retained_at_mean_median),
            output / "fig3b_after_median_gate.png", 520)
    values = completeness.complete_count
    fig, ax = plt.subplots(figsize=(12, 9))
    sns.histplot(values, bins=15, kde=True, color="#1f77b4", alpha=0.8, ax=ax)
    ax.axvline(values.median(), color="#c0392b", linestyle="--", linewidth=2,
               label=f"Median = {values.median():g}")
    ax.set_xlabel("Mean complete-prompt count out of 520", fontsize=30)
    ax.set_ylabel("Count", fontsize=30)
    ax.tick_params(labelsize=24)
    ax.legend(fontsize=24)
    fig.tight_layout()
    fig.savefig(output / "fig3c_distribution.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    pairs = pd.read_csv(release / "discovery/pair_metrics.csv")
    for letter, (slug, model) in zip("abcd", MODELS.items()):
        frame = pairs.loc[pairs.model.eq(model)].rename(columns={
            "mutator_1": "m1", "mutator_2": "m2", "matched_prompts": "complete_count"}).copy()
        frame["conditional_asr"] = 100 * frame.chain_successes / frame.complete_count.replace(0, np.nan)
        heatmap(matrix(frame, "conditional_asr", frame.average_gate_matched_success),
                output / f"fig4{letter}_{slug}.png", 100)

    overlap = pd.read_csv(release / "derived/cross_target_overlap.csv").sort_values("jaccard")
    labels = [f"{SHORT_NAMES[r.model_1]}–{SHORT_NAMES[r.model_2]}" for r in overlap.itertuples()]
    fig, ax = plt.subplots(figsize=(6.4, 4.7))
    positions = np.arange(len(overlap))
    ax.barh(positions, overlap.jaccard, height=0.42, color="#31688e",
            edgecolor="black", linewidth=0.9, zorder=3)
    for y, value in zip(positions, overlap.jaccard):
        ax.text(value + 0.025, y, f"{value:.3f}", va="center", fontsize=13)
    ax.set_yticks(positions, labels, fontsize=13)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Jaccard similarity\n(0 = no overlap, 1 = identical sets)", fontsize=13)
    ax.xaxis.grid(True, linestyle=":", alpha=0.55)
    ax.set_axisbelow(True)
    for side in ["top", "right", "left"]:
        ax.spines[side].set_visible(False)
    fig.tight_layout(pad=0.3)
    fig.savefig(output / "fig_jaccard_dotplot.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True, sharey=True)
    for ax, kind in zip(axes, ["intent", "persistence"]):
        data = pd.read_csv(release / f"derived/{kind}_validation.csv")
        data = data.loc[data.judge.ne("majority")]
        for row, name, marker, color in zip(data.itertuples(),
                ["GPT-4o Mini", "Qwen", "Gemma"], ["o", "s", "^"],
                ["#31688e", "#35b779", "#d95f02"]):
            ax.scatter(row.precision * 100, row.recall * 100, s=80,
                       label=name, marker=marker, color=color, zorder=3)
        ax.plot([0, 100], [0, 100], color="gray", linestyle="--", linewidth=1)
        ax.set_xlabel("Precision (%)")
        ax.set_title(f"{kind.capitalize()} judge")
        ax.set_xlim(65, 100)
        ax.set_ylim(65, 100)
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("Recall (%)")
    fig.tight_layout()
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center",
               bbox_to_anchor=(0.5, 1.12), ncol=3)
    fig.savefig(output / "judge_precision_recall.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Generated nine empirical panels in {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", type=Path, default=DEFAULT_RELEASE)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/figures"))
    args = parser.parse_args()
    plot(args.release, args.output_dir)


if __name__ == "__main__":
    main()
