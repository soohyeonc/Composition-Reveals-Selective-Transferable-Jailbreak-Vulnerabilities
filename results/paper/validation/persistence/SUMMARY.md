# Persistence-Classifier Validation

The existing human annotation contains 132 ordered-pair chains. Each chain contributes one M1 label and one M2 label, giving 264 decisions. M1 compares the original prompt with the final chained prompt. M2 compares the intermediate prompt with the final chained prompt.

The labels are imbalanced. M1 contains 39 positive and 93 negative labels, while M2 contains 125 positive and 7 negative labels. Balanced accuracy and the confusion counts are therefore essential alongside raw accuracy.

## Overall Results

| Judge | Accuracy | Balanced accuracy | Precision | Recall | F1 | Cohen kappa | TN / FP / FN / TP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| GPT-4o Mini | 91.67% [88.26%, 94.70%] | 91.54% | 94.38% | 92.07% | 93.21% | 0.824 | 91 / 9 / 13 / 151 |
| Qwen3.8 27B | 85.61% [81.44%, 89.39%] | 87.44% | 96.32% | 79.88% | 87.33% | 0.710 | 95 / 5 / 33 / 131 |
| Gemma 4 31B | 88.64% [84.85%, 92.05%] | 88.32% | 91.88% | 89.63% | 90.74% | 0.760 | 87 / 13 / 17 / 147 |
| Three-judge majority | 89.77% [86.36%, 93.18%] | 90.21% | 94.77% | 88.41% | 91.48% | 0.787 | 92 / 8 / 19 / 145 |

## Results by Chain Position

| Position | Judge | Human positive | Accuracy | Balanced accuracy | TN / FP / FN / TP |
| --- | --- | ---: | ---: | ---: | ---: |
| M1 | GPT-4o Mini | 39/132 | 90.91% | 90.57% | 85 / 8 / 4 / 35 |
| M1 | Qwen3.8 27B | 39/132 | 88.64% | 83.75% | 89 / 4 / 11 / 28 |
| M1 | Gemma 4 31B | 39/132 | 84.85% | 83.29% | 81 / 12 / 8 / 31 |
| M1 | Three-judge majority | 39/132 | 88.64% | 85.98% | 86 / 7 / 8 / 31 |
| M2 | GPT-4o Mini | 125/132 | 92.42% | 89.26% | 6 / 1 / 9 / 116 |
| M2 | Qwen3.8 27B | 125/132 | 82.58% | 84.06% | 6 / 1 / 22 / 103 |
| M2 | Gemma 4 31B | 125/132 | 92.42% | 89.26% | 6 / 1 / 9 / 116 |
| M2 | Three-judge majority | 125/132 | 90.91% | 88.46% | 6 / 1 / 11 / 114 |

## Results by Mutator

Each mutator has 22 labels. These accuracies are diagnostic because the per-mutator samples are small.

| Mutator | Human positive | GPT-4o Mini | Qwen3.8 27B | Gemma 4 31B | Majority |
| --- | ---: | ---: | ---: | ---: | ---: |
| ea-encryption | 11/22 | 95.45% | 95.45% | 100.00% | 95.45% |
| ea-fictional | 15/22 | 86.36% | 95.45% | 68.18% | 86.36% |
| ea-obfuscation | 12/22 | 100.00% | 100.00% | 100.00% | 100.00% |
| ea-paraphrasing | 13/22 | 86.36% | 77.27% | 81.82% | 86.36% |
| ea-translation | 16/22 | 100.00% | 100.00% | 100.00% | 100.00% |
| mm-cognitive-hacking | 18/22 | 77.27% | 72.73% | 72.73% | 77.27% |
| mm-forced-completion | 9/22 | 95.45% | 95.45% | 100.00% | 100.00% |
| mm-gaslighting | 13/22 | 90.91% | 54.55% | 77.27% | 77.27% |
| mm-nshot-hacking | 13/22 | 95.45% | 95.45% | 95.45% | 95.45% |
| mm-privilege-escalation | 16/22 | 90.91% | 77.27% | 95.45% | 86.36% |
| mm-prompt-injection | 13/22 | 86.36% | 81.82% | 86.36% | 86.36% |
| mm-roleplay | 15/22 | 95.45% | 81.82% | 86.36% | 86.36% |

## Pairwise Judge Agreement

| Judge 1 | Judge 2 | Agreement | Cohen kappa |
| --- | --- | ---: | ---: |
| GPT-4o Mini | Qwen3.8 27B | 87.12% | 0.741 |
| GPT-4o Mini | Gemma 4 31B | 90.91% | 0.810 |
| Qwen3.8 27B | Gemma 4 31B | 89.39% | 0.787 |

## Interpretation Boundary

This is a retrospective validation on the study's existing annotations, not a newly sampled or independently held-out human study. Each mutator has 22 labels overall and only 11 labels in each position, so per-mutator estimates are diagnostic rather than precise. The source labels may also have informed development of the current evaluator prompts.

The validation directly tests the persistence classifier used to form completeness-based screens. It does not alter raw all-520 target-model ASR, which does not depend on persistence labels.

Machine-readable overall, position, mutator, mutator-position, agreement, disagreement, and error tables are stored beside this summary.
