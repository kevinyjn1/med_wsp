# EXP-03 Results - MIT experiment 8: the Qwen3.5-0.8B math edit, rebuilt and checked on this Mac (parts A and B)

Part C (our own Astra session on Qwen) is paused with 0 forward passes used and is not covered here.

## Experiment
Replication of experiment 8 of the MIT CSAIL post "Sparks of Weight-Space Programming": Astra edited
Qwen3.5-0.8B so that it stops looping while thinking and solves more math problems. Part A asks whether MIT's
edited checkpoint can be rebuilt exactly from public weights and their released diff. Part B asks whether that
edit changes behaviour on this Mac in the direction MIT reports. No formal experiment contract was written. The
working specification is `vault/experiments/MIT replication spec (exp 3, 7, 8).md`, section 3, and
`vault/experiments/EXP-03 MIT exp 8 - editing Qwen3.5-0.8B for math.md`. The scale of part B (48 problems, one
sample, 16,384-token limit, MLX) is Claude's provisional choice, made because MIT's full evaluation is out of
reach locally.

## Run status
COMPLETE for parts A and B as scoped. Part B is a scaled-down check, not a replication of MIT's evaluation.

## What actually ran
### Part A - rebuild
- `python experiments/exp_03_mit8_math/apply_edit.py --run p50c` applies MIT's released diff for the 50-pass run
  to `Qwen/Qwen3.5-0.8B` revision `2fc06364715b967f1860aea9cf38778875588b17` and compares SHA-256 hashes of the
  base and edited tensors with the hashes in MIT's manifest. Ran 2026-11-22.

### Part B - behaviour check
- `eval_mlx.py`, once with the unedited model and once with the rebuilt p50c edit.
- Back end: MLX (`mlx-lm` 0.32.2, `mlx` 0.32.4), edited tensors rounded to bfloat16 like a saved checkpoint.
- Problems: 48 from the GSM8K test split, sampled with seed 0; the same 48 for both models.
- Prompt and sampling as in MIT's evaluation: chat template with thinking enabled, the problem followed by the
  instruction line, temperature 0.6, top-p 0.95, top-k 20, presence penalty 1.5. One sample per problem, sampling
  seed 0, batches of 12.
- Limit: 16,384 new tokens per answer (MIT: 32,768).
- Grading as in MIT's evaluation: the content of the last `\boxed{...}` after the thinking block, checked with
  math-verify; no closed box counts as wrong.
- Per answer we also store: whether it hit the token limit ("cut"), whether the thinking block was closed, and a
  repetition measure (share of the last 512 tokens covered by an 8-token sequence that occurred earlier in the
  answer).
- Unedited model: 2026-11-23 09:59 to 12:45. Edited model: 12:45 to 13:19.

## Deviations from contract
- No contract exists.
- From MIT's evaluation protocol (all by design of the scaled-down check): 48 problems instead of 1,319; one
  sample per problem; 16,384 instead of 32,768 new tokens; MLX on Apple silicon instead of vLLM on a rented GPU.
- The unedited run's process was suspended by the experimenter from 12:14 to 12:33 (the Mac was on battery). The
  Mac was also asleep during that run from 10:07 to 11:17 (battery empty, hibernation) and with the lid closed
  from 11:40 to 11:59, 12:00 to 12:12 and 12:26 to 12:32 (`pmset -g log`). The process continued from the same
  state each time.

## Artifacts
- `processed/edit_p50c_report.json` - part A: per tensor, hash checks, changed entries and columns, relative
  change, share of the change in its first singular direction, recovered strength.
- `processed/eval_base_gsm8k_n48_t16384.jsonl`, `processed/eval_p50c_gsm8k_n48_t16384.jsonl` - part B: one line
  per problem (gold, prediction, correct, tokens, cut, closed thinking, repetition, first and last 600 characters).
- `processed/eval_*_gsm8k_n48_t16384.summary.json` - the two summaries.
- `work/edit_p50c/` - the rebuilt edited tensors (local only, not committed).
- Logs (not committed): `logs/exp03_eval_base_gsm8k.log`, `logs/exp03_eval_p50c_gsm8k.log`.
- Viewer dataset `mit8_qwen_p50c` (the edit itself; no dataset for part B).

## Numerical results
### Part A
| Tensor | Base hash | Edited hash | Entries changed (manifest) | Columns changed | Relative change | Share of change in one direction | Recovered strength |
|---|---|---|---|---|---|---|---|
| `layers.15.mlp.down_proj.weight` (1,024 x 3,584) | matches | matches | 65,260 (65,260) | 64 | 0.10636 | 0.9999945 | 6.00032 |
| `layers.19.mlp.down_proj.weight` (1,024 x 3,584) | matches | matches | 64,928 (64,928) | 64 | 0.08106 | 0.9999914 | 6.00017 |

Cosine between the two layers' directions: 0.8490.

### Part B (48 GSM8K problems, limit 16,384 tokens)
| | Unedited | MIT's p50c edit |
|---|---|---|
| Correct | 12 of 48 (25.0 %) | 25 of 48 (52.1 %) |
| Hit the token limit | 36 of 48 (75.0 %) | 15 of 48 (31.3 %) |
| Closed the thinking block | 11 of 48 (22.9 %) | 34 of 48 (70.8 %) |
| No boxed answer | 25 | 15 |
| Boxed answer given but wrong | 11 | 8 |
| Mean tokens per answer | 14,143 | 10,181 |
| Median tokens per answer | 16,384 | 11,279 |
| Mean repetition measure | 0.867 | 0.534 |
| Correct among answers that finished | 11 of 12 | 25 of 33 |
| Correct among answers that hit the limit | 1 of 36 | 0 of 15 |

Paired over the 48 problems: 14 wrong without the edit and right with it, 1 right without and wrong with it, 11
right in both. Accuracy difference +27.1 points (standard error 7.1), exact two-sided sign test p = 0.00098.
Token limit: 22 problems hit it only without the edit, 1 only with the edit, 14 in both runs.

For reference, MIT's published numbers for the same edit on all 1,319 GSM8K test problems with a 32,768-token
limit: accuracy 13.8 % to 67.1 %, share at the token limit 87.5 % to 7.2 %, mean tokens 29,289 to 11,993.

## Observations
- `[OBSERVATION]` Part A: the checkpoint MIT evaluated is reproduced bit for bit from the public base weights and
  the released diff. The change in each matrix is `W' = W - c u (u^T W)` on 64 columns with c = 6.000 and more
  than 99.999 % of its energy in one direction.
- `[OBSERVATION]` Part B: with the edit, accuracy on these 48 problems rises from 25.0 % to 52.1 % (14 fixed, 1
  broken, p = 0.001).
- `[OBSERVATION]` With the edit, fewer answers run to the token limit (36 to 15 of 48), more close the thinking
  block (11 to 34), and the repetition measure falls from 0.867 to 0.534.
- `[OBSERVATION]` In both runs almost no answer that hits the limit is correct (1 of 36, 0 of 15). Among answers
  that finish, 11 of 12 are correct without the edit and 25 of 33 with it.
- `[OBSERVATION]` The direction of every effect matches MIT's report. The sizes differ (accuracy 25.0 % to
  52.1 % here against 13.8 % to 67.1 %; limit hits 75.0 % to 31.3 % against 87.5 % to 7.2 %), with a different
  token limit, sample and back end.

## Negative / null results
- None.

## Failed / excluded runs
- The first launch of the sweep queue (2026-11-22 23:31) failed for both models before loading anything (shell
  quoting error, exit 127). The log files of that attempt were overwritten by this run.
- Part C, first attempt `p10_local` (2026-11-22 17:10): stopped by the experimenter after 3.6 hours with 0 forward
  passes used; its first generation pass asked for 64 answers of up to 16,384 tokens and did not finish. Raw
  record in `raw/p10_local/`.

## Anomalies
- Wall time is not a measure of speed here. The unedited run took 9,943 s and the edited run 2,079 s, but the
  unedited run includes the suspension and the sleeps listed above and shared the GPU with the Astra donor
  session, and its answers are longer.
- One unedited answer (`gsm8k-202`) is graded correct although it hit the token limit without closing the
  thinking block: it repeats "I'll make sure to use `\boxed{100}`" until the limit, and the grader takes the last
  closed box.

## Scope
- One model (Qwen3.5-0.8B), one edit (MIT's p50c), one benchmark sample (48 GSM8K test problems), one sample per
  problem, one sampling seed, 16,384-token limit, MLX.

## What this experiment does NOT establish
- It does not reproduce MIT's numbers. The sample is 48 of 1,319 problems and the token limit is half of MIT's,
  so answers between 16,384 and 32,768 tokens count as failures here.
- It says nothing about MATH-500, IFEval or the other benchmarks MIT reports, and nothing about MIT's 10-pass and
  25-pass edits.
- It does not show that Astra can find such an edit in our harness: part C has not produced a single pass.
- With one sample per problem at temperature 0.6, it does not separate the edit's effect on one problem from
  sampling noise; only the paired total over 48 problems is tested.
- It does not show that the suspension of the unedited run left its output unchanged; that run was not repeated.

## Reproducibility
- Commits: `apply_edit.py` last changed in `6310308`, `eval_mlx.py` and `wsp/mathqa.py` in `fc725bc`. The
  unedited run started on the tree at `98112a8`, the edited run on `69a1948`; the evaluation code is the same in
  both.
- Commands:
  `python experiments/exp_03_mit8_math/apply_edit.py --run p50c`
  `python experiments/exp_03_mit8_math/eval_mlx.py --model base --set gsm8k --n 48 --max-tokens 16384 --batch-size 12`
  `python experiments/exp_03_mit8_math/eval_mlx.py --model p50c --set gsm8k --n 48 --max-tokens 16384 --batch-size 12`
- Seeds: 0 for the problem sample, 0 for sampling (`mx.random.seed(0)` once per run; answers depend on the batch
  composition).
- Needs MIT's released logs in `external/lm_understand-logs/exp8_math/` and the base model in the local
  Hugging Face cache.

## Status
Awaiting researcher review. Not promoted to `research/07_RESULTS_GROUND_TRUTH.md`.
