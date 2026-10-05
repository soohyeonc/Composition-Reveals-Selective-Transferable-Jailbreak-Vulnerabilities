# Intent-Judge Validation on StrongREJECT Human Evaluations

The primary reference is whether the median human jailbreak-effectiveness score is nonzero. This is a compatible proxy for attempted intent fulfillment, not an exact human annotation of the paper's binary intent definition.

The dataset contains 1,361 prompt-response pairs, 39 unique forbidden prompts, and up to five human ratings per pair.

## Primary Results

| Judge | Accuracy | Balanced accuracy | Precision | Recall | F1 | Cohen kappa | Confusion (TN/FP/FN/TP) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| GPT-4o Mini | 89.79% [87.43%, 92.01%] | 81.62% | 88.48% | 65.95% | 75.57% | 0.693 | 1007/28/111/215 |
| Qwen3.8 27B | 94.05% [92.35%, 95.67%] | 89.36% | 93.91% | 80.37% | 86.61% | 0.828 | 1018/17/64/262 |
| Gemma 4 31B | 94.49% [93.20%, 95.79%] | 93.43% | 86.38% | 91.41% | 88.82% | 0.852 | 988/47/28/298 |
| Three-judge majority | 94.42% [92.87%, 95.91%] | 89.92% | 94.64% | 81.29% | 87.46% | 0.839 | 1020/15/61/265 |

## Pairwise Judge Agreement

| Judge 1 | Judge 2 | Agreement | Cohen kappa |
| --- | --- | ---: | ---: |
| GPT-4o Mini | Qwen3.8 27B | 93.09% | 0.777 |
| GPT-4o Mini | Gemma 4 31B | 88.98% | 0.677 |
| Qwen3.8 27B | Gemma 4 31B | 94.27% | 0.838 |

## Interpretation Boundary

StrongREJECT humans rated jailbreak effectiveness on a five-point scale. The study's intent judge instead asks whether a response attempts to fulfill the original request even when incorrect or incomplete. Agreement therefore measures external convergent validity. It must not be described as exact intent-classification accuracy.

Threshold sensitivity, source-model breakdowns, judge disagreement rows, and machine-readable confusion counts are included beside this summary.
