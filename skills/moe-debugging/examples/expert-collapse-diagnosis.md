# Worked example: diagnosing router collapse in a 16-expert run

## Symptom

> "Training has been stuck at loss ~2.9 since around step 4000 of 5000.
> Step times look fine, but the loss won't come down. I suspect something
> wrong with the experts."

Training logs show no OOM, no error traces, and steady step times. The only
signal is a plateau: loss fell quickly in the first few hundred steps, then
leveled off and has not moved in ~1000 steps.

## Data collected

**Config:** 16 experts per layer, top-2 routing, aux loss `0.001`,
capacity factor `1.0`, BF16, gradient clipping `1.0`, Adam with 200-step
warmup.

**Router statistics** — per-expert token counts over one step window
(`router_counts.csv`):

```csv
expert,count
0,550
1,250
2,14
3,14
4,14
5,14
6,14
7,14
8,14
9,14
10,14
11,14
12,14
13,14
14,14
15,18
```

**Loss curve** — 60-step excerpt of the run around step 5000 (`loss.csv`;
only the head and tail of the 60 rows are shown):

```csv
step,loss
0,3.620
1,3.210
2,2.780
3,2.340
4,2.020
5,1.884
6,2.310
7,2.620
8,2.780
9,2.860
...
48,2.904
49,2.911
50,2.904
51,2.911
52,2.907
53,2.913
54,2.902
55,2.908
56,2.914
57,2.905
58,2.904
59,2.911
```

Two experts (0 and 1) take 80% of the tokens (550 + 250 of 1000). The loss
bottomed at 1.884 around step 5 and has hovered near 2.9 ever since.

## Analyzer runs

### Router distribution

```text
$ python3 skills/moe-debugging/analyzers/router_distribution.py \
    --input router_counts.csv

Router distribution analysis
============================
  total tokens         : 1000
  experts (n)          : 16
  entropy (normalized) : 0.5499
  gini coefficient     : 0.6970
  top-expert share     : 0.5500
  effective experts    : 4.59
  utilization skew     : 8.80
  overflow fraction    : 0.1250
FLAGS: COLLAPSED IMBALANCED OVERFLOW
```

### Loss analysis

```text
$ python3 skills/moe-debugging/analyzers/loss_analyzer.py --input loss.csv

Loss curve analysis
===================
  steps        : 60
  min loss     : 1.8840
  max loss     : 3.6200
  final loss   : 2.9110
  nan count    : 0
  inf count    : 0
  spike count  : 0
FLAGS: PLATEAU

Flag details
------------
  NAN        : 0 rows
  INF        : 0 rows
  SPIKE      : 0 steps (|loss - median(11)| > 5*MAD(11))
  PLATEAU    : slope=0.00000/step (last 50 rows), final 2.9110 > 1.5*min 1.8840 -> True
  DIVERGENCE : last 20 finite rows monotonic non-decreasing -> False
```

### Expert utilization

```text
$ python3 skills/moe-debugging/analyzers/expert_utilization.py \
    --input router_counts.csv

Expert utilization analysis
===========================
  total tokens           : 1000
  experts (n)            : 16
  utilization %          : 100.0%
  skew (max/min)         : 39.29
  overflow experts       : 2
  overflow token fraction: 0.8000
FLAGS: OVERFLOW
```

### Diagnosis report

```text
$ python3 skills/moe-debugging/tools/diagnosis_report.py \
    --router router_counts.csv --loss loss.csv --expert router_counts.csv
```

Output:

```markdown
# MoE Training Diagnosis Report

## 1. Router distribution

| Metric | Value |
| --- | --- |
| Total tokens | 1,000 |
| Experts (n) | 16 |
| Normalized entropy | 0.5499 |
| Gini coefficient | 0.6970 |
| Top-expert share | 0.5500 |
| Effective experts | 4.59 |
| Utilization skew | 8.80 |
| Overflow fraction | 0.1250 |

## 2. Loss analysis

| Metric | Value |
| --- | --- |
| Steps | 60 |
| Min loss | 1.8840 |
| Max loss | 3.6200 |
| Final loss | 2.9110 |
| NaN rows | 0 |
| Inf rows | 0 |
| Spike count | 0 |
| Top spike steps | [] |
| Plateau | True |
| Divergence | False |

## 3. Expert utilization

| Metric | Value |
| --- | --- |
| Total tokens | 1,000 |
| Experts (n) | 16 |
| Utilization % | 100.0% |
| Skew (max/min) | 39.29 |
| Overflow experts | 2 |
| Overflow token fraction | 0.8000 |

| Expert | Count | Share % | Capacity |
| --- | --- | --- | --- |
| e00 | 550 | 55.0% | OVER |
| e01 | 250 | 25.0% | OVER |
| e02 | 14 | 1.4% | OK |
| e03 | 14 | 1.4% | OK |
| e04 | 14 | 1.4% | OK |
| e05 | 14 | 1.4% | OK |
| e06 | 14 | 1.4% | OK |
| e07 | 14 | 1.4% | OK |
| e08 | 14 | 1.4% | OK |
| e09 | 14 | 1.4% | OK |
| e10 | 14 | 1.4% | OK |
| e11 | 14 | 1.4% | OK |
| e12 | 14 | 1.4% | OK |
| e13 | 14 | 1.4% | OK |
| e14 | 14 | 1.4% | OK |
| e15 | 18 | 1.8% | OK |

## 4. Synthesis

All flags observed across analyzers:

- `COLLAPSED`
- `IMBALANCED`
- `OVERFLOW`
- `PLATEAU`

**Strongest signal:** Router collapse.
**Recommendation:** apply the `### Router collapse` workflow in SKILL.md.

_capacity-factor used: 1; diagnosis is evidence-driven, not a training change._
```

## Diagnosis

### Problem

A 16-expert, top-2 MoE has plateaued at loss ~2.9 for the last ~1000 steps.
Step times are healthy, so the failure is in routing, not compute or
communication.

### Evidence

- **`COLLAPSED`** — top-expert share 0.55 (expert 0 alone takes 55% of
  tokens), which is the literal collapse threshold.
- **`IMBALANCED`** — Gini 0.697 (far above the 0.3 threshold) and only
  4.59 effective experts out of 16 (well below `0.5 × 16 = 8`).
- **`OVERFLOW`** — 12.5% of experts sit above their balanced capacity of 62.5
  tokens, holding 80% of all tokens; at capacity factor 1.0, tokens are being
  dropped from experts 0 and 1 every step.
- **`PLATEAU`** — final loss 2.911 is above `1.5 × min = 2.826`, with a
  zero-slope tail. Numerical flags are absent (`NAN` 0, `INF` 0, `SPIKE` 0),
  ruling out exploding-loss/NaN as the cause.

### Likely causes (ranked)

1. **Aux loss too low (0.001).** The load-balancing term is a rounding error
   against the task loss, so nothing stops the router from concentrating on
   experts 0 and 1.
2. **Capacity factor 1.0 under imbalance.** Tokens from the two overloaded
   experts are dropped every step; the dropped tokens lose their expert
   contribution, which reads on the curve as a plateau.
3. **No jitter.** The router deterministically favors the same experts,
   so the 14 under-loaded experts receive almost no gradient and stay
   dormant.

### Recommended actions

1. Raise aux loss `0.001` → `0.01` (top of the 0.001–0.01 range) to push the
   router back toward uniform load.
2. Raise capacity factor `1.0` → `1.25` so overloaded experts stop dropping
   tokens while the aux loss rebalances them.
3. Add jitter noise to router logits at train time so dormant experts receive
   gradient.

### Ablation plan

Train a short run (e.g. 2k steps, fixed seed) with aux loss `0.01` vs `0`
and jitter on vs off. Confirm collapse is fixed when the aux `0.01` + jitter
run holds `effective experts ≥ 8` (re-check with `router_distribution.py`)
and the loss resumes its descent instead of re-plateauing.

## Resolution summary

With aux loss at `0.01`, capacity factor `1.25`, and jitter on, the router
stops concentrating on two experts: the load-balancing term actively pulls
tokens toward the 14 under-loaded experts, the extra capacity stops the token
drops while that happens, and jitter keeps the dormant experts in the
gradient path. Expected effects: effective experts rises from 4.59 toward the
healthy `≥ 8` band, Gini falls below 0.3, `OVERFLOW` clears, and the loss
resumes decreasing past the 1.884 minimum. The ablation (aux 0.01 vs 0, fixed
seed) is what confirms the diagnosis before a full re-training.
