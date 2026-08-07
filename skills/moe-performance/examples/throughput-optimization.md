# Worked example: throughput optimization for a 7B-MoE run

## Input

> "Throughput is poor — 40% GPU util, 3.2 s/step on 8×H100. Improve
> throughput."

Config summary of the run under analysis (from the `moe-architecture` /
`moe-training` docs):

| Field | Value |
| --- | --- |
| Total params | 7.0 B (dense 1.0 B, expert-only 6.0 B) |
| Experts / top-k | 64 / 2 |
| Layers, d_model, vocab, seq_len | 24, 2048, 32000, 2048 |
| Expert FFN hidden | 32768 |
| Capacity factor | 1.25 |
| Aux loss | 0.001 (weak) |
| Parallelism | DP=2 × EP=4 (16 experts/rank), TP=1, PP=1 |
| Batch geometry | micro-batch 8, grad accum 20, global batch 320 |
| Hardware | 8 × H100-80GB, one node, NVLink |

Tokens per step: `320 × 2048 = 655,360`.

## Baseline

Run `profilers/throughput_profiler.py` on a 30-step window. Inputs (excerpts):

```text
# steps.csv (step,seconds,busy)
step,seconds,busy
1,3.1,1.24
2,3.2,1.28
3,2.9,1.16
4,3.3,1.32
5,3.1,1.24
...
29,3.4,1.36
30,3.1,1.24
```

```text
# experts.csv (expert,count) — one step window, 64 experts
expert,count
0,22000
1,10500
2,3500
3,3000
4,2500
5,998
...
62,998
63,998
```

Invocation:

```text
python3 skills/moe-performance/profilers/throughput_profiler.py \
    --steps steps.csv --expert experts.csv \
    --tokens-per-step 655360 --gpus 8 \
    --flops-per-token 7.6e9 --capacity-factor 1.25
```

Output:

```text
MoE throughput baseline
=======================
  steps                : 30
  step time (s)        : mean 3.200  p50 3.200  p95 3.400  min 2.900  max 3.500
  tokens/sec (global)  : 204,800.0
  tokens/sec per GPU   : 25,600.0
  expert utilization % : 80.0%  (capacity factor 1.25)
  expert skew (max/min): 22.04
  top-expert share %   : 21.9%
  GPU util proxy %     : 40.0%  (busy/seconds)
  bubble time %        : 60.0%
  MFU % (ROUGH EST.)   : 19.7%

Baseline CSV:
  baseline.csv
```

`baseline.csv`:

```text
metric,value,unit
tokens_per_sec_global,204800,tokens/s
tokens_per_sec_per_gpu,25600,tokens/s/gpu
p50_step_s,3.2,s
p95_step_s,3.4,s
gpu_util_proxy,40,%
expert_util_pct,80,%
mfu_pct,19.6724,%
bubble_pct,60,%
```

Notes: `--flops-per-token 7.6e9` is the FLOPs/token from the architecture
skill for this model (~1.2B activated params). MFU is a rough estimate; what
matters is the shape: MFU (19.7%) sits well below the util proxy (40%), so the
busy time itself is not spent on useful matmuls.

## Analysis

Walking the 7-step workflow:

1. **Baseline** — 204,800 tokens/sec global (25,600/GPU), 3.2 s/step, 40%
   util, 60% bubble. For a 7B-MoE on 8×H100 this is roughly half the
   throughput the hardware should deliver.
2. **Expert utilization** — Gini **0.350**, top-expert share **21.9%**, skew
   22.0. The router is heavily imbalanced: one expert takes 22% of tokens and
   many take ~1%. Expert utilization reads 80%, but that is just the
   balanced-capacity cap at cf 1.25 (`100/1.25`); the real cost is that hot
   experts overflow capacity and drop tokens while cold experts sit idle.
3. **Communication** — EP=4, top-2, cf 1.25. Dispatch volume per MoE layer ≈
   `top_k × tokens × 2 B = 2 × 655,360 × 2 ≈ 2.6 MB` per direction, × 24
   layers ≈ 63 MB/step per direction. On NVLink that is a few milliseconds per
   layer, but with no overlap it is a meaningful share of a 3.2 s step. The
   comm-side lever to test is EP 4→8 (finer sharding, more fan-out) and the
   capacity factor (smaller buffers).
4. **Memory** — run the `moe-training` estimator on the DP=2/EP=4 layout:

   ```text
   python3 skills/moe-training/tools/memory_estimator.py --total-params-b 7.0 \
       --expert-params-b 6.0 --precision bf16 --optimizer adamw --dp 2 --tp 1 \
       --pp 1 --ep 4 --gpus 8 --num-experts 64 --micro-batch-size 8 \
       --seq-len 2048 --d-model 2048 --num-layers 24 --gpu-mem-gb 80
   ```

   ```text
   Per-GPU memory budget (bf16 precision, adamw optimizer, DP=2 TP=1 PP=1 EP=4, 8 GPUs)
   +--------------------+------------+
   | Item               |         GB |
   +--------------------+------------+
   | Parameters         |    4.00 GB |
   | Gradients          |    4.00 GB |
   | Optimizer states   |   24.00 GB |
   | Activations        |   16.11 GB |
   | Overhead+buffers   |    6.31 GB |
   | TOTAL              |   54.42 GB |
   +--------------------+------------+

     GPU memory limit:    80.00 GB
     Headroom:             32.0%  (25.58 GB free)
   ```

   Memory is **OK**: ≈68% used, 32% headroom. Not the binding constraint.
5. **Kernel/bottleneck signals** — micro-batch 8 gives short expert matmuls
   (each expert processes ~2,048 tokens/micro-batch), and padding to cf 1.25
   under the skew wastes compute. Both point to utilization, not memory.
6. **Rank** — see the Ranked plan section.
7. **Verify** — each gain below has an A/B step-time check; none assumes the
   others hold, and gains overlap, so treat the combined estimate as
   conservative.

## Ranked plan

Candidates and their inputs to `tools/bottleneck_rank.py`:

```text
# candidates.csv (bottleneck,impact,probability,cost)
bottleneck,impact,probability,cost
ep_degree,0.13,0.60,2.0
load_balancing_aux_loss,0.16,0.60,2.0
capacity_factor,0.10,0.60,3.0
sequence_packing,0.25,0.60,4.0
kernel_fusion,0.20,0.30,7.0
small_micro_batch,0.08,0.50,5.0
```

Invocation and output:

```text
python3 skills/moe-performance/tools/bottleneck_rank.py --input candidates.csv --top 6

Ranked bottlenecks (by ROI = impact * probability / cost)
==========================================================
rank | bottleneck                   | impact |  prob | cost |     ROI |      gain band
--------------------------------------------------------------------------------------
   1 | load_balancing_aux_loss      |   0.16 |  0.60 |  2.0 |  0.0480 |       4.8-9.6%
   2 | ep_degree                    |   0.13 |  0.60 |  2.0 |  0.0390 |       3.9-7.8%
   3 | sequence_packing             |   0.25 |  0.60 |  4.0 |  0.0375 |      7.5-15.0%
   4 | capacity_factor              |   0.10 |  0.60 |  3.0 |  0.0200 |       3.0-6.0%
   5 | kernel_fusion                |   0.20 |  0.30 |  7.0 |  0.0086 |       3.0-6.0%
   6 | small_micro_batch            |   0.08 |  0.50 |  5.0 |  0.0080 |       2.0-4.0%
```

(Gain band = `impact × probability`, an ESTIMATE of the tokens/sec gain
fraction; low is half the high.)

Top optimizations, in rank order:

1. **Load-balancing aux loss 0.001 → 0.01** — expected gain +4.8–9.6% util
   (headline ~+6–10%). Risk: over-strengthening distorts routing. Verify: Gini
   and top-expert share before/after, plus dropped-token count.
2. **EP 4 → 8** — expected gain +3.9–7.8% tokens/sec (headline ~+5–8%). Risk:
   all-to-all fan-out grows (EP=8 means an 8-way exchange per MoE layer).
   Verify: A/B step time at EP=8 vs EP=4 on the same batch geometry.
3. **Sequence packing** — expected gain +7.5–15% tokens/sec (headline
   ~+10–15%). Risk: packing implementation and attention masking. Verify: util
   and step time with packing on vs off.
4. **Capacity factor 1.25 → 1.0** — expected gain +3.0–6.0% tokens/sec,
   and −4% expert compute from removed padding. Risk: dropped tokens if the
   router is still imbalanced — only safe after the aux-loss step. Verify:
   drop count stays near zero, expert utilization stays near 100%.
5. **Kernel fusion** — +3–6% on compute-bound regions, but **future scope**:
   requires Triton/CUDA kernels profiled with Nsight. Not actionable now.

## Baseline vs proposed

Conservative combined estimate: the gains overlap, so apply ~25% of the
headline tokens/sec gain rather than compounding the full bands.

| Metric | Baseline | Proposed | Delta |
| --- | --- | --- | --- |
| tokens/sec (global) | 204,800 | ~256,000 | +25% |
| Step time (s) | 3.2 | ~2.56 | −20% |
| GPU util proxy (%) | 40 | ~50 | +10 pp |
| Expert utilization (%) | 80 | ~100¹ | +20 pp |
| MFU (ROUGH EST.) (%) | 19.7 | ~24.6 | +25% |
| Bubble time (%) | 60 | ~50 | −10 pp |

¹ Expert utilization uses the balanced-capacity definition, so it is capped at
`100 / capacity_factor`; the jump to 100% reflects moving to cf 1.0, not a
quality gain — the real win is that a balanced router drops no tokens at the
lower capacity.

## Verification plan

Run each change as a fixed-seed A/B on 30+ steps and compare with the same
profiler invocation as the baseline:

| Change | Metric that decides | Target |
| --- | --- | --- |
| Aux loss 0.001 → 0.01 | Gini, top-expert share, dropped tokens | Gini < 0.2, top share < 8%, drops ≈ 0 |
| EP 4 → 8 | step time at same global batch | step time ≥ 5% lower |
| Sequence packing on | GPU util proxy, step time | util +10 pp, step time −10% |
| Capacity 1.25 → 1.0 | dropped-token count | drops ≈ 0 with balanced router |

Confirm combined gains by re-running the profiler once with all four changes
applied; the combined tokens/sec should land near the 256,000 estimate.
