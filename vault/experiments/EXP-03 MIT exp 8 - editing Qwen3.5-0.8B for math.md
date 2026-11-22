---
type: experiment
id: EXP-03
status: running
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

## Part B - behaviour of the edited model on this Mac (running)
MIT evaluated with vLLM on a rented GPU, up to 32,768 new tokens per answer; the unedited model runs to the limit
on 93 % of MATH-500 answers, and the median correct edited answer is 9,172 tokens long.
- Speed measured here (other jobs running at the same time): PyTorch on MPS 8 tokens/s single and 43 at batch 16
  (the linear-attention layers fall back to reference code); MLX (`mlx-lm` 0.32.2) 53 single, 393 at batch 16,
  600 at batch 64.
- Scaled-down check with MLX (`eval_mlx.py`): 64 GSM8K test problems, one sample each, same prompt and sampling,
  16,384-token cap, base vs MIT's p50c edit. Queued 2026-11-22; results in `processed/eval_*_gsm8k_n64_t16384.summary.json`.
- MIT's full evaluation volume (about 61.7M tokens for unedited MATH-500 alone) stays out of reach locally.

## Part C - our own Astra session (running)
`python experiments/exp_03_mit8_math/run_astra.py --name p10_local --budget 10 --effort max`, started
2026-11-22 17:10. Lab: `wsp/qwen_lab.py`. Differences from MIT's harness: at most 64 answers per generation pass
(32 per model with "both") and 16,384 new tokens; teacher-forced passes limited to 256 texts / 100,000 tokens; no
"activations" option on generation passes; original and edited models are sampled independently; explanations
required from the first turn.

## Visualizations
Viewer dataset `mit8_qwen_p50c`: the two edited matrices before and after, the change, the 64 edited columns
and the loop direction. See [[Visualization index]].

## Links
[[Weight-space programming]] - [[Astra harness]] - [[2026-11-21 Session 01]]
