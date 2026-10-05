# Third-party material

The root MIT license applies to original project code and documentation that
the contributors are entitled to license. It does not relicense third-party
datasets, externally supplied prompts, model weights, model outputs or papers.
Retain upstream notices when redistributing those materials.

## AdvBench

The 520 original harmful-behavior requests in the frozen prompt export originate
from [llm-attacks/AdvBench](https://github.com/llm-attacks/llm-attacks/tree/main/data/advbench).
The upstream repository provides the following MIT notice in its
[license file](https://github.com/llm-attacks/llm-attacks/blob/main/LICENSE).

MIT License

Copyright (c) 2023 Andy Zou

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## StrongREJECT

Intent-judge validation uses the [StrongREJECT human-evaluation release](https://osf.io/download/jwmqe/).
The curated artifact redistributes row identifiers, human median scores,
judge labels and opaque prompt-cluster identifiers, not the source
prompt-response corpus. The cluster identifiers preserve the original grouping
for bootstrap confidence intervals. The original source hash is retained for
matching. The optional inference runner downloads the source and verifies that
hash. Obtain the corpus and its terms from its authors.
The [StrongREJECT code repository](https://github.com/dsbowen/strong_reject)
has its own MIT license, copyright 2024 Dillon Bowen. Do not infer dataset or
model-output rights solely from a code license.

## Other research material

The model families, jailbreak strategies and evaluation rubrics are attributed
in the manuscript. Model weights are not redistributed in the curated release.
Frozen GPT-3.5 mutations and paper-aligned
target-response texts are included for reproducibility. Their original SHA-256
hashes remain in the immutable release; response text is stored separately in
`release/paper-support/`. The root MIT license does not purport to grant rights in
model outputs or third-party text embedded in them. These security-research
inputs and outputs contain harmful material. Handle them accordingly.
