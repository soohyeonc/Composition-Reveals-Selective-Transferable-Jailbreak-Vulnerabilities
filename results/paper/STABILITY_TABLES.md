# Reported seven-condition stability panels

Every cell uses 520 prompts. The rate is pair ASR. Parentheses give pair successes vs the stronger standalone component, then their multiplier. A zero baseline gives n/a. These are descriptive selected-panel measurements, not confidence intervals.

## GPT-5.6 Luna

| Role | Ordered pair | Original | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Positive | Cognitive hacking → N-shot hacking | **7.50%**<br>(39 vs 5; 7.80x) | **7.12%**<br>(37 vs 6; 6.17x) | **7.88%**<br>(41 vs 5; 8.20x) | **7.50%**<br>(39 vs 5; 7.80x) | **7.12%**<br>(37 vs 5; 7.40x) | **7.50%**<br>(39 vs 6; 6.50x) |
| Positive | N-shot hacking → Obfuscation | **4.62%**<br>(24 vs 5; 4.80x) | **4.81%**<br>(25 vs 6; 4.17x) | **4.62%**<br>(24 vs 5; 4.80x) | **4.23%**<br>(22 vs 5; 4.40x) | **4.04%**<br>(21 vs 5; 4.20x) | **5.00%**<br>(26 vs 6; 4.33x) |
| Positive | Gaslighting → N-shot hacking | **4.81%**<br>(25 vs 5; 5.00x) | **4.81%**<br>(25 vs 6; 4.17x) | **5.19%**<br>(27 vs 5; 5.40x) | **3.65%**<br>(19 vs 5; 3.80x) | **3.65%**<br>(19 vs 5; 3.80x) | **3.27%**<br>(17 vs 6; 2.83x) |
| Positive | N-shot hacking → Translation | **3.08%**<br>(16 vs 5; 3.20x) | **3.85%**<br>(20 vs 6; 3.33x) | **4.62%**<br>(24 vs 5; 4.80x) | **3.65%**<br>(19 vs 5; 3.80x) | **3.46%**<br>(18 vs 5; 3.60x) | **3.65%**<br>(19 vs 6; 3.17x) |
| Positive | Fictional scenario → Paraphrasing | **2.88%**<br>(15 vs 10; 1.50x) | **2.69%**<br>(14 vs 9; 1.56x) | **3.46%**<br>(18 vs 10; 1.80x) | **2.69%**<br>(14 vs 9; 1.56x) | **3.08%**<br>(16 vs 10; 1.60x) | **3.65%**<br>(19 vs 13; 1.46x) |
| Reverse | N-shot hacking → Cognitive hacking | **0.19%**<br>(1 vs 5; 0.20x) | **0.00%**<br>(0 vs 6; 0.00x) | **0.00%**<br>(0 vs 5; 0.00x) | **0.19%**<br>(1 vs 5; 0.20x) | **0.19%**<br>(1 vs 5; 0.20x) | **0.19%**<br>(1 vs 6; 0.17x) |
| Negative | Cognitive hacking → Obfuscation | **0.00%**<br>(0 vs 0; n/a) | **0.00%**<br>(0 vs 1; 0.00x) | **0.00%**<br>(0 vs 0; n/a) | **0.00%**<br>(0 vs 1; 0.00x) | **0.00%**<br>(0 vs 0; n/a) | **0.00%**<br>(0 vs 1; 0.00x) |

## DeepSeek R1 8B

| Role | Ordered pair | Original | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Positive | Forced completion → Encryption | **13.08%**<br>(68 vs 15; 4.53x) | **13.46%**<br>(70 vs 14; 5.00x) | **12.69%**<br>(66 vs 12; 5.50x) | **11.92%**<br>(62 vs 15; 4.13x) | **13.85%**<br>(72 vs 12; 6.00x) | **12.69%**<br>(66 vs 16; 4.12x) |
| Positive | Privilege escalation → Encryption | **11.92%**<br>(62 vs 15; 4.13x) | **9.04%**<br>(47 vs 14; 3.36x) | **10.00%**<br>(52 vs 12; 4.33x) | **9.04%**<br>(47 vs 15; 3.13x) | **8.27%**<br>(43 vs 12; 3.58x) | **10.38%**<br>(54 vs 16; 3.38x) |
| Positive | Cognitive hacking → N-shot hacking | **9.81%**<br>(51 vs 15; 3.40x) | **9.81%**<br>(51 vs 17; 3.00x) | **8.46%**<br>(44 vs 15; 2.93x) | **9.23%**<br>(48 vs 14; 3.43x) | **10.00%**<br>(52 vs 12; 4.33x) | **9.04%**<br>(47 vs 11; 4.27x) |
| Positive | Forced completion → Fictional scenario | **32.69%**<br>(170 vs 108; 1.57x) | **32.12%**<br>(167 vs 113; 1.48x) | **30.58%**<br>(159 vs 108; 1.47x) | **32.50%**<br>(169 vs 121; 1.40x) | **30.19%**<br>(157 vs 113; 1.39x) | **29.42%**<br>(153 vs 117; 1.31x) |
| Positive | Fictional scenario → Encryption | **29.62%**<br>(154 vs 108; 1.43x) | **32.31%**<br>(168 vs 113; 1.49x) | **28.85%**<br>(150 vs 108; 1.39x) | **30.00%**<br>(156 vs 121; 1.29x) | **30.77%**<br>(160 vs 113; 1.42x) | **30.38%**<br>(158 vs 117; 1.35x) |
| Reverse | Fictional scenario → Forced completion | **0.77%**<br>(4 vs 108; 0.04x) | **1.35%**<br>(7 vs 113; 0.06x) | **0.38%**<br>(2 vs 108; 0.02x) | **1.15%**<br>(6 vs 121; 0.05x) | **0.96%**<br>(5 vs 113; 0.04x) | **0.58%**<br>(3 vs 117; 0.03x) |
| Negative | Cognitive hacking → Obfuscation | **0.00%**<br>(0 vs 1; 0.00x) | **0.19%**<br>(1 vs 0; n/a) | **0.00%**<br>(0 vs 0; n/a) | **0.19%**<br>(1 vs 0; n/a) | **0.00%**<br>(0 vs 0; n/a) | **0.19%**<br>(1 vs 0; n/a) |

## Gemma 4 31B

| Role | Ordered pair | Original | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Positive | Gaslighting → N-shot hacking | **5.38%**<br>(28 vs 7; 4.00x) | **4.81%**<br>(25 vs 6; 4.17x) | **5.38%**<br>(28 vs 6; 4.67x) | **5.19%**<br>(27 vs 5; 5.40x) | **5.38%**<br>(28 vs 5; 5.60x) | **5.58%**<br>(29 vs 5; 5.80x) |
| Positive | Cognitive hacking → N-shot hacking | **5.00%**<br>(26 vs 7; 3.71x) | **4.23%**<br>(22 vs 6; 3.67x) | **5.58%**<br>(29 vs 6; 4.83x) | **4.42%**<br>(23 vs 5; 4.60x) | **4.23%**<br>(22 vs 5; 4.40x) | **4.23%**<br>(22 vs 5; 4.40x) |
| Positive | Gaslighting → Forced completion | **3.27%**<br>(17 vs 9; 1.89x) | **3.85%**<br>(20 vs 8; 2.50x) | **4.04%**<br>(21 vs 7; 3.00x) | **4.23%**<br>(22 vs 8; 2.75x) | **3.46%**<br>(18 vs 8; 2.25x) | **3.65%**<br>(19 vs 9; 2.11x) |
| Positive | Translation → Paraphrasing | **2.12%**<br>(11 vs 3; 3.67x) | **1.15%**<br>(6 vs 3; 2.00x) | **1.54%**<br>(8 vs 5; 1.60x) | **1.73%**<br>(9 vs 3; 3.00x) | **1.15%**<br>(6 vs 3; 2.00x) | **1.73%**<br>(9 vs 4; 2.25x) |
| Positive | Forced completion → Translation | **3.65%**<br>(19 vs 9; 2.11x) | **2.69%**<br>(14 vs 8; 1.75x) | **3.46%**<br>(18 vs 7; 2.57x) | **3.65%**<br>(19 vs 8; 2.38x) | **2.12%**<br>(11 vs 8; 1.38x) | **3.08%**<br>(16 vs 9; 1.78x) |
| Reverse | N-shot hacking → Gaslighting | **0.19%**<br>(1 vs 7; 0.14x) | **0.58%**<br>(3 vs 6; 0.50x) | **0.19%**<br>(1 vs 6; 0.17x) | **0.19%**<br>(1 vs 5; 0.20x) | **0.19%**<br>(1 vs 5; 0.20x) | **0.00%**<br>(0 vs 5; 0.00x) |
| Negative | Cognitive hacking → Obfuscation | **0.00%**<br>(0 vs 0; n/a) | **0.00%**<br>(0 vs 1; 0.00x) | **0.00%**<br>(0 vs 0; n/a) | **0.00%**<br>(0 vs 1; 0.00x) | **0.00%**<br>(0 vs 0; n/a) | **0.19%**<br>(1 vs 0; n/a) |

## Qwen3.8 27B

| Role | Ordered pair | Original | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Positive | Gaslighting → N-shot hacking | **1.73%**<br>(9 vs 1; 9.00x) | **2.31%**<br>(12 vs 1; 12.00x) | **2.31%**<br>(12 vs 1; 12.00x) | **2.12%**<br>(11 vs 1; 11.00x) | **1.73%**<br>(9 vs 1; 9.00x) | **2.31%**<br>(12 vs 2; 6.00x) |
| Positive | Cognitive hacking → N-shot hacking | **1.73%**<br>(9 vs 1; 9.00x) | **1.35%**<br>(7 vs 1; 7.00x) | **1.73%**<br>(9 vs 1; 9.00x) | **1.15%**<br>(6 vs 1; 6.00x) | **1.35%**<br>(7 vs 1; 7.00x) | **1.73%**<br>(9 vs 2; 4.50x) |
| Positive | N-shot hacking → Translation | **0.38%**<br>(2 vs 1; 2.00x) | **1.35%**<br>(7 vs 1; 7.00x) | **1.15%**<br>(6 vs 2; 3.00x) | **0.77%**<br>(4 vs 3; 1.33x) | **1.54%**<br>(8 vs 1; 8.00x) | **0.96%**<br>(5 vs 2; 2.50x) |
| Positive | N-shot hacking → Paraphrasing | **0.58%**<br>(3 vs 1; 3.00x) | **0.96%**<br>(5 vs 1; 5.00x) | **0.58%**<br>(3 vs 1; 3.00x) | **0.77%**<br>(4 vs 1; 4.00x) | **0.19%**<br>(1 vs 1; 1.00x) | **0.77%**<br>(4 vs 2; 2.00x) |
| Positive | Gaslighting → Translation | **0.58%**<br>(3 vs 1; 3.00x) | **0.96%**<br>(5 vs 1; 5.00x) | **0.38%**<br>(2 vs 2; 1.00x) | **0.19%**<br>(1 vs 3; 0.33x) | **0.38%**<br>(2 vs 0; n/a) | **0.19%**<br>(1 vs 1; 1.00x) |
| Reverse | N-shot hacking → Gaslighting | **0.00%**<br>(0 vs 1; 0.00x) | **0.00%**<br>(0 vs 1; 0.00x) | **0.00%**<br>(0 vs 1; 0.00x) | **0.00%**<br>(0 vs 1; 0.00x) | **0.00%**<br>(0 vs 1; 0.00x) | **0.00%**<br>(0 vs 2; 0.00x) |
| Negative | Cognitive hacking → Obfuscation | **0.00%**<br>(0 vs 0; n/a) | **0.00%**<br>(0 vs 0; n/a) | **0.00%**<br>(0 vs 0; n/a) | **0.00%**<br>(0 vs 0; n/a) | **0.00%**<br>(0 vs 0; n/a) | **0.00%**<br>(0 vs 0; n/a) |
