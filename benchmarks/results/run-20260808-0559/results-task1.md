# Results — task1 (run-20260808-0559)

Metric: correctness (total /25) (0-25). n=5 seeds per arm.

| Arm | n | mean | SD | values |
|---|---|---|---|---|
| A0 | 5 | 21 | 2.449 | [21, 25, 21, 19, 19] |
| A1 | 5 | 21 | 1.414 | [23, 21, 21, 19, 21] |
| A2 | 5 | 22.6 | 2.191 | [21, 21, 25, 25, 21] |
| A3 | 5 | 22.2 | 2.683 | [21, 25, 21, 19, 25] |

**A3 vs A1 (primary comparison for this task):** Cohen's d = 0.447, paired t-test p = 0.3739

### Cost (this task)
| Arm | mean tokens | mean tool calls | mean wall-clock (s) |
|---|---|---|---|
| A0 | 14100.8 | 0.6 | 50.1 |
| A1 | 19951 | 0.6 | 92.9 |
| A2 | 54991.2 | 2.6 | 64.4 |
| A3 | 67858.2 | 3 | 120.7 |

**Ceiling-effect caveat:** the A1-only headroom check (n=2, see headroom-check.md) found scores of 21/25 and 25/25 — already near the rubric maximum. This task has reduced statistical power to detect an A3-vs-A1 effect and results here should be read with that in mind, per methodology.md's pre-registered ceiling-effect handling.
