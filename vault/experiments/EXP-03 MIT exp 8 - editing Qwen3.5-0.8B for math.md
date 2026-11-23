---
type: experiment
id: EXP-03
status: parts A and B done (awaiting review), part C paused
tags: [experiment, mit-replication, astra, weight-edit]
---

# EXP-03 - MIT exp 8: editing Qwen3.5-0.8B for math

## Question
MIT's Astra made Qwen3.5-0.8B (thinking on) much better at math by subtracting a "loop direction" from 64 MLP
output columns in each of layers 15 and 19 (experiment 8 of [[Takehi et al. 2026 - Weight-Space Programming]]).
Can we (a) reproduce their edited model exactly and see the same behaviour, and (b) have our own Astra session
rediscover such an edit?

## Setup
- Model: `Qwen/Qwen3.5-0.8B`, revision `2fc06364715b967f1860aea9cf38778875588b17` (the one MIT edited).
- MIT's released edit diffs: `external/lm_understand-logs/exp8_math/runs/<run>/edit/`.
- Protocol details: [[MIT replication spec (exp 3, 7, 8)]], section 3.

## Part A - rebuild MIT's edit (done)
`python experiments/exp_03_mit8_math/apply_edit.py --run p50c` applies the released diff to our local base
weights and checks every hash. Result (`experiments/exp_03_mit8_math/processed/edit_p50c_report.json`):

| Tensor | Base sha256 | Rebuilt sha256 | Entries changed | Columns | Relative change | Rank-1 share of the change | Recovered strength |
|---|---|---|---|---|---|---|---|
| layers.15.mlp.down_proj | matches MIT | matches MIT | 65,260 | 64 | 0.1064 | 0.999995 | 6.0003 |
| layers.19.mlp.down_proj | matches MIT | matches MIT | 64,928 | 64 | 0.0811 | 0.999991 | 6.0002 |

- The edit is `W' = W - 6 u (u^T W)` restricted to 64 columns per layer, as MIT's logs describe.
- The two layers use different directions: cosine 0.849.
- So the edited checkpoint MIT evaluated is reproduced bit for bit from public weights.

## Part B - behaviour of the edited model on this Mac (done 2026-11-23)
MIT evaluated with vLLM on a rented GPU, up to 32,768 new tokens per answer; the unedited model runs to the limit
on 93 % of MATH-500 answers, and the median correct edited answer is 9,172 tokens long.
- Speed measured here (other jobs running at the same time): PyTorch on MPS 8 tokens/s single and 43 at batch 16
  (the linear-attention layers fall back to reference code); MLX (`mlx-lm` 0.32.2) 53 single, 393 at batch 16,
  600 at batch 64.
- Scaled-down check with MLX (`eval_mlx.py`): 48 GSM8K test problems, one sample each, same prompt and sampling,
  16,384-token cap, base vs MIT's p50c edit. Ran 2026-11-23 09:59 to 13:19; results in
  `processed/eval_*_gsm8k_n48_t16384.summary.json`, handoff in `experiments/exp_03_mit8_math/RESULTS.md`.

| 48 GSM8K problems, limit 16,384 tokens | Unedited | MIT's p50c edit |
|---|---|---|
| Correct | 12 (25.0 %) | 25 (52.1 %) |
| Hit the token limit | 36 (75.0 %) | 15 (31.3 %) |
| Closed the thinking block | 11 | 34 |
| Mean tokens per answer | 14,143 | 10,181 |
| Repetition measure | 0.867 | 0.534 |

- `[OBSERVATION]` Paired: 14 problems fixed, 1 broken, 11 right in both (+27.1 points, SE 7.1, p = 0.001).
- `[OBSERVATION]` Every effect goes the way MIT reports (GSM8K 13.8 % to 67.1 %, limit hits 87.5 % to 7.2 % on
  1,319 problems with a 32,768-token limit). The sizes are not comparable: 48 problems, half the token limit, MLX.
- Not established: MIT's numbers themselves, any other benchmark, or that Astra can find such an edit here.
- MIT's full evaluation volume (about 61.7M tokens for unedited MATH-500 alone) stays out of reach locally.

## Part C - our own Astra session (paused)
First attempt `p10_local` (started 2026-11-22 17:10, effort max): its first generation pass asked for 64 answers of
up to 16,384 tokens; under MLX at batch 32 that grew to 17 GB and had not finished after 3.6 hours on a machine
that was already swapping. Stopped with 0 passes used. Changes since: at most 32 answers per pass (16 per model
with "both"), worker batch 16. Not restarted yet: the MedQA work has priority on the GPU, and the Part B behaviour
check (now 48 problems, batch 12) is queued behind it.

## Visualizations
Viewer dataset `mit8_qwen_p50c`: the two edited matrices before and after, the change, the 64 edited columns
and the loop direction. See [[Visualization index]].

## Links
[[Weight-space programming]] - [[Astra harness]] - [[2026-11-21 Session 01]]
