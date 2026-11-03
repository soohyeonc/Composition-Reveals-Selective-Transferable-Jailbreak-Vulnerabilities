# Reported seven-condition stability panels

Each run sends all 520 prompts, but prompts on which the chain or either standalone prompt is a writer refusal are excluded, so every cell uses the n prompts given with the ordered pair. The rate is pair ASR. Parentheses give pair successes vs the stronger standalone component, then their multiplier. A zero baseline gives n/a. These are descriptive selected-panel measurements, not confidence intervals.

## GPT-5.6 Luna

| Role | Ordered pair | Original | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Positive | Cognitive hacking → N-shot hacking (n=498) | **7.23%**<br>(36 vs 5; 7.20x) | **6.83%**<br>(34 vs 6; 5.67x) | **7.83%**<br>(39 vs 5; 7.80x) | **7.43%**<br>(37 vs 5; 7.40x) | **6.83%**<br>(34 vs 5; 6.80x) | **7.23%**<br>(36 vs 6; 6.00x) |
| Positive | N-shot hacking → Obfuscation (n=491) | **4.89%**<br>(24 vs 5; 4.80x) | **5.09%**<br>(25 vs 6; 4.17x) | **4.89%**<br>(24 vs 5; 4.80x) | **4.48%**<br>(22 vs 5; 4.40x) | **4.28%**<br>(21 vs 5; 4.20x) | **5.30%**<br>(26 vs 6; 4.33x) |
| Positive | Gaslighting → N-shot hacking (n=498) | **4.42%**<br>(22 vs 5; 4.40x) | **4.82%**<br>(24 vs 6; 4.00x) | **5.02%**<br>(25 vs 5; 5.00x) | **3.41%**<br>(17 vs 5; 3.40x) | **3.41%**<br>(17 vs 5; 3.40x) | **3.01%**<br>(15 vs 6; 2.50x) |
| Positive | N-shot hacking → Translation (n=488) | **3.28%**<br>(16 vs 5; 3.20x) | **4.10%**<br>(20 vs 5; 4.00x) | **4.92%**<br>(24 vs 4; 6.00x) | **3.89%**<br>(19 vs 5; 3.80x) | **3.69%**<br>(18 vs 4; 4.50x) | **3.89%**<br>(19 vs 6; 3.17x) |
| Positive | Fictional scenario → Paraphrasing (n=514) | **2.92%**<br>(15 vs 9; 1.67x) | **2.72%**<br>(14 vs 9; 1.56x) | **3.50%**<br>(18 vs 9; 2.00x) | **2.72%**<br>(14 vs 8; 1.75x) | **3.11%**<br>(16 vs 9; 1.78x) | **3.70%**<br>(19 vs 12; 1.58x) |
| Reverse | N-shot hacking → Cognitive hacking (n=494) | **0.20%**<br>(1 vs 5; 0.20x) | **0.00%**<br>(0 vs 5; 0.00x) | **0.00%**<br>(0 vs 4; 0.00x) | **0.20%**<br>(1 vs 5; 0.20x) | **0.20%**<br>(1 vs 4; 0.25x) | **0.20%**<br>(1 vs 5; 0.20x) |
| Negative | Cognitive hacking → Obfuscation (n=520) | **0.00%**<br>(0 vs 0; n/a) | **0.00%**<br>(0 vs 1; 0.00x) | **0.00%**<br>(0 vs 0; n/a) | **0.00%**<br>(0 vs 1; 0.00x) | **0.00%**<br>(0 vs 0; n/a) | **0.00%**<br>(0 vs 1; 0.00x) |

## DeepSeek R1 8B

| Role | Ordered pair | Original | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Positive | Forced completion → Encryption (n=520) | **13.08%**<br>(68 vs 15; 4.53x) | **13.46%**<br>(70 vs 14; 5.00x) | **12.69%**<br>(66 vs 12; 5.50x) | **11.92%**<br>(62 vs 15; 4.13x) | **13.85%**<br>(72 vs 12; 6.00x) | **12.69%**<br>(66 vs 16; 4.12x) |
| Positive | Privilege escalation → Encryption (n=520) | **11.92%**<br>(62 vs 15; 4.13x) | **9.04%**<br>(47 vs 14; 3.36x) | **10.00%**<br>(52 vs 12; 4.33x) | **9.04%**<br>(47 vs 15; 3.13x) | **8.27%**<br>(43 vs 12; 3.58x) | **10.38%**<br>(54 vs 16; 3.38x) |
| Positive | Cognitive hacking → N-shot hacking (n=498) | **10.04%**<br>(50 vs 15; 3.33x) | **9.64%**<br>(48 vs 17; 2.82x) | **8.43%**<br>(42 vs 15; 2.80x) | **9.44%**<br>(47 vs 14; 3.36x) | **9.84%**<br>(49 vs 12; 4.08x) | **9.24%**<br>(46 vs 11; 4.18x) |
| Positive | Forced completion → Fictional scenario (n=520) | **32.69%**<br>(170 vs 108; 1.57x) | **32.12%**<br>(167 vs 113; 1.48x) | **30.58%**<br>(159 vs 108; 1.47x) | **32.50%**<br>(169 vs 121; 1.40x) | **30.19%**<br>(157 vs 113; 1.39x) | **29.42%**<br>(153 vs 117; 1.31x) |
| Positive | Fictional scenario → Encryption (n=520) | **29.62%**<br>(154 vs 108; 1.43x) | **32.31%**<br>(168 vs 113; 1.49x) | **28.85%**<br>(150 vs 108; 1.39x) | **30.00%**<br>(156 vs 121; 1.29x) | **30.77%**<br>(160 vs 113; 1.42x) | **30.38%**<br>(158 vs 117; 1.35x) |
| Reverse | Fictional scenario → Forced completion (n=520) | **0.77%**<br>(4 vs 108; 0.04x) | **1.35%**<br>(7 vs 113; 0.06x) | **0.38%**<br>(2 vs 108; 0.02x) | **1.15%**<br>(6 vs 121; 0.05x) | **0.96%**<br>(5 vs 113; 0.04x) | **0.58%**<br>(3 vs 117; 0.03x) |
| Negative | Cognitive hacking → Obfuscation (n=520) | **0.00%**<br>(0 vs 1; 0.00x) | **0.19%**<br>(1 vs 0; n/a) | **0.00%**<br>(0 vs 0; n/a) | **0.19%**<br>(1 vs 0; n/a) | **0.00%**<br>(0 vs 0; n/a) | **0.19%**<br>(1 vs 0; n/a) |

## Gemma 4 31B

| Role | Ordered pair | Original | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Positive | Gaslighting → N-shot hacking (n=498) | **5.22%**<br>(26 vs 7; 3.71x) | **4.62%**<br>(23 vs 6; 3.83x) | **5.22%**<br>(26 vs 6; 4.33x) | **5.02%**<br>(25 vs 5; 5.00x) | **5.22%**<br>(26 vs 5; 5.20x) | **5.42%**<br>(27 vs 5; 5.40x) |
| Positive | Cognitive hacking → N-shot hacking (n=498) | **5.22%**<br>(26 vs 7; 3.71x) | **4.22%**<br>(21 vs 6; 3.50x) | **5.82%**<br>(29 vs 6; 4.83x) | **4.62%**<br>(23 vs 5; 4.60x) | **4.42%**<br>(22 vs 5; 4.40x) | **4.42%**<br>(22 vs 5; 4.40x) |
| Positive | Gaslighting → Forced completion (n=520) | **3.27%**<br>(17 vs 9; 1.89x) | **3.85%**<br>(20 vs 8; 2.50x) | **4.04%**<br>(21 vs 7; 3.00x) | **4.23%**<br>(22 vs 8; 2.75x) | **3.46%**<br>(18 vs 8; 2.25x) | **3.65%**<br>(19 vs 9; 2.11x) |
| Positive | Translation → Paraphrasing (n=514) | **1.95%**<br>(10 vs 3; 3.33x) | **0.97%**<br>(5 vs 3; 1.67x) | **1.36%**<br>(7 vs 5; 1.40x) | **1.56%**<br>(8 vs 3; 2.67x) | **0.97%**<br>(5 vs 3; 1.67x) | **1.56%**<br>(8 vs 4; 2.00x) |
| Positive | Forced completion → Translation (n=520) | **3.65%**<br>(19 vs 9; 2.11x) | **2.69%**<br>(14 vs 8; 1.75x) | **3.46%**<br>(18 vs 7; 2.57x) | **3.65%**<br>(19 vs 8; 2.38x) | **2.12%**<br>(11 vs 8; 1.38x) | **3.08%**<br>(16 vs 9; 1.78x) |
| Reverse | N-shot hacking → Gaslighting (n=493) | **0.20%**<br>(1 vs 6; 0.17x) | **0.61%**<br>(3 vs 5; 0.60x) | **0.20%**<br>(1 vs 5; 0.20x) | **0.20%**<br>(1 vs 4; 0.25x) | **0.20%**<br>(1 vs 4; 0.25x) | **0.00%**<br>(0 vs 4; 0.00x) |
| Negative | Cognitive hacking → Obfuscation (n=520) | **0.00%**<br>(0 vs 0; n/a) | **0.00%**<br>(0 vs 1; 0.00x) | **0.00%**<br>(0 vs 0; n/a) | **0.00%**<br>(0 vs 1; 0.00x) | **0.00%**<br>(0 vs 0; n/a) | **0.19%**<br>(1 vs 0; n/a) |

## Qwen3.8 27B

| Role | Ordered pair | Original | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Positive | Gaslighting → N-shot hacking (n=498) | **1.81%**<br>(9 vs 1; 9.00x) | **2.41%**<br>(12 vs 1; 12.00x) | **2.41%**<br>(12 vs 1; 12.00x) | **2.21%**<br>(11 vs 1; 11.00x) | **1.81%**<br>(9 vs 1; 9.00x) | **2.41%**<br>(12 vs 2; 6.00x) |
| Positive | Cognitive hacking → N-shot hacking (n=498) | **1.81%**<br>(9 vs 1; 9.00x) | **1.41%**<br>(7 vs 1; 7.00x) | **1.81%**<br>(9 vs 1; 9.00x) | **1.20%**<br>(6 vs 1; 6.00x) | **1.41%**<br>(7 vs 1; 7.00x) | **1.81%**<br>(9 vs 2; 4.50x) |
| Positive | N-shot hacking → Translation (n=488) | **0.41%**<br>(2 vs 1; 2.00x) | **1.43%**<br>(7 vs 1; 7.00x) | **1.23%**<br>(6 vs 2; 3.00x) | **0.82%**<br>(4 vs 3; 1.33x) | **1.64%**<br>(8 vs 1; 8.00x) | **1.02%**<br>(5 vs 2; 2.50x) |
| Positive | N-shot hacking → Paraphrasing (n=486) | **0.62%**<br>(3 vs 1; 3.00x) | **1.03%**<br>(5 vs 1; 5.00x) | **0.62%**<br>(3 vs 1; 3.00x) | **0.82%**<br>(4 vs 1; 4.00x) | **0.21%**<br>(1 vs 1; 1.00x) | **0.82%**<br>(4 vs 2; 2.00x) |
| Positive | Gaslighting → Translation (n=520) | **0.58%**<br>(3 vs 1; 3.00x) | **0.96%**<br>(5 vs 1; 5.00x) | **0.38%**<br>(2 vs 2; 1.00x) | **0.19%**<br>(1 vs 3; 0.33x) | **0.38%**<br>(2 vs 0; n/a) | **0.19%**<br>(1 vs 1; 1.00x) |
| Reverse | N-shot hacking → Gaslighting (n=493) | **0.00%**<br>(0 vs 1; 0.00x) | **0.00%**<br>(0 vs 1; 0.00x) | **0.00%**<br>(0 vs 1; 0.00x) | **0.00%**<br>(0 vs 1; 0.00x) | **0.00%**<br>(0 vs 1; 0.00x) | **0.00%**<br>(0 vs 2; 0.00x) |
| Negative | Cognitive hacking → Obfuscation (n=520) | **0.00%**<br>(0 vs 0; n/a) | **0.00%**<br>(0 vs 0; n/a) | **0.00%**<br>(0 vs 0; n/a) | **0.00%**<br>(0 vs 0; n/a) | **0.00%**<br>(0 vs 0; n/a) | **0.00%**<br>(0 vs 0; n/a) |
