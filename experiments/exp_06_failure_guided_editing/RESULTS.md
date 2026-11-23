# EXP-06 Results - Failure-guided weight editing of Phi-3.5 on MedQA (no donor), run1

## Experiment
Filippos Bellos's "Experiment 1" (2026-11-21): give GPT-6 Astra a general model's wrong MedQA answers and
interpretability tools, and ask it to repair the responsible weights directly, with no donor model and no
fine-tuning. No formal experiment contract was written. The working specification is the vault note
`vault/experiments/EXP-06 Failure-guided weight editing.md` and the task text sent to Astra
(`raw/run1/message.txt`); the budget, the effort level and the edit restrictions are Claude's provisional
choices, not researcher decisions.

## Run status
COMPLETE. Astra used all 30 forward passes and submitted no edit.

## What actually ran
- Model edited: `microsoft/Phi-3.5-mini-instruct` (float16 on Apple MPS). Editor: `gpt-6-astra` through
  `codex exec`, effort `xhigh`, all Codex tools off, every turn audited as tool-free.
- Astra's material: the weights in a sandboxed CPU-only Python kernel, 4,272 practice questions (MedQA dev 1,272
  + the first 3,000 training questions) with answers and the unedited model's result on each, and 30 forward
  passes (score passes of up to 1,024 questions with paired statistics, or teacher-forced reads of activations).
- Prompt for every score: the chat prompt of EXP-04 Protocol A (instruction line, question, options A-D,
  "Answer:"; the answer is the letter with the highest next-token probability).
- Session: 69 replies from Astra, 36 kernel runs, 31 forward requests of which 30 were counted (23 score passes
  over 4,992 question slots, 7 teacher-forced reads). One conversation restart with the notebook at reply 41.
  Started 2026-11-22 16:39, interrupted at pass 20, resumed 21:45, final answer 23:06. Astra's own thinking
  time: 2,483 s. Time inside tools: 3.23 h.
- Astra's own protocol (its choice, recorded in `raw/run1/work/splits.json` and `candidate_lock.json` before the
  final passes): it split the practice pool at random into discovery 1,200, selection 1,024, development reserve
  1,024 and final confirmation 1,024, and fixed a retention rule before looking at the confirmation set: keep
  the edit only if the paired accuracy gain on the 1,024 confirmation questions exceeds 2 standard errors and
  the exact two-sided sign test gives p < 0.05.
- Final measurement by the harness: the submitted weights vs the unedited model on the MedQA test split
  (1,273 questions, never shown to Astra), paired.

## Deviations from contract
- No contract exists.
- Tool error for passes 1 to 20: `hidden_states` layer 32 was the residual stream after the final RMSNorm, not
  before it as the task text said. Astra detected this itself (notes, after pass 7) and corrected its analysis.
  From pass 21 on, layer 32 is the pre-norm residual. Astra was told through `notices/001_tool_corrections.txt`.
- The session was interrupted at 20 of 30 passes by a machine restart (the lab process had grown to 18 GB) and
  resumed from saved state. The Python kernel's variables were lost; files in the work folder were kept.
- From pass 21 on, every forward pass runs in a fresh worker process (`wsp/lab_worker.py`).
- Pass 21 was requested three times. The first attempt was killed with the lab process, the second ended with
  `WorkerError` because the experimenter stopped the worker to load corrected code. Neither was counted against
  the budget and neither returned any result to Astra.

## Artifacts
- `raw/run1/message.txt` - the task text. `raw/run1/turns/tNNN/` - every prompt, reply and Codex event stream.
- `raw/run1/tool_calls.jsonl`, `raw/run1/forward_calls.jsonl` - every kernel run and forward request with output.
- `raw/run1/final_answer.md` - Astra's final answer. `raw/run1/final_eval.json`,
  `raw/run1/final_eval_per_question.jsonl` - the harness's test-split measurement.
- `raw/run1/notices/001_tool_corrections.txt` - the note delivered at reply 48.
- Viewer dataset `exp06_edit_run1` (`python -m wsp.viz_export exp06`; data under `viz/public/data/`, not in git);
  screenshots `vault/attachments/viz_exp06_run1_unit.jpg`, `viz_exp06_run1_per_question.jpg`.
- Check of the saved candidate by Claude (`work/locked_l26_orth_minus3/`): only column 4131 differs from the
  original tensor; the kept part lies in the span of the three centered readout directions of the tokens
  "_A".."_D" (residual 9e-08) and the reversed part is orthogonal to it (1e-07); the column's norm goes from 2.55
  to 7.50; the tensor's relative change is 4.66 %.
- Not in git (the `work` folder is ignored; it stays on this machine at `raw/run1/work/`): `notes.md` (Astra's
  notebook), `final_results.json`, `splits.json`, `candidate_lock.json`, `score_<k>.jsonl` (per-question results
  of each score pass), `fwd_<k>/` (activation arrays), and three saved candidate snapshots.

## Numerical results

### Final measurement (MedQA test split, n = 1,273, chat prompt)
| | accuracy | predicted A / B / C / D |
|---|---|---|
| unedited Phi-3.5 | 0.5719 | 252 / 324 / 364 / 333 |
| submitted weights | 0.5719 | 252 / 324 / 364 / 333 |

Edited tensors: none. Wrong-to-right 0, right-to-wrong 0. The submitted weights are the original weights.

### The candidate Astra locked and then rejected
Edit: in `model.layers.26.mlp.down_proj.weight`, column 4131 (the output direction of one MLP unit): keep the
part of the column that lies in the span of the answer-letter readout contrasts, and multiply the rest by -3.
One column of one tensor. SHA-256 of the edited tensor: `7ca4e416...6f537c` (`candidate_lock.json`).

Paired results, edited vs original, on practice questions (accuracy in percentage points, pp):

| question set | role | n | original | edited | difference | SE | wrong->right | right->wrong | sign test p |
|---|---|---|---|---|---|---|---|---|---|
| selection, first 512 (passes 21+22) | used to choose the edit | 512 | 56.64 % | 58.59 % | +1.95 pp | 0.73 | 12 | 2 | 0.013 |
| development reserve, first 256 (pass 26) | independent | 256 | 60.94 % | 60.16 % | -0.78 pp | 0.78 | 1 | 3 | 0.625 |
| final confirmation (passes 27-30) | independent, locked beforehand | 1,024 | 57.52 % | 58.20 % | +0.68 pp | 0.54 | 19 | 12 | 0.281 |

- The four confirmation batches of 256, in order: +1.95, -0.39, -0.39, +1.56 pp.
- Mean change of the gold-answer margin on the confirmation set: +0.32 (SE 0.16) log-prob units.
- Both independent sets pooled (Astra's figure, 1,280 questions): +0.39 pp, SE 0.46.
- Astra's retention rule (gain > 2 SE and sign test p < 0.05) was not met. It restored the original weights.
- Re-count by Claude from the lab's own per-question files `score_27..30.jsonl`: 589 and 596 correct of 1,024,
  19 wrong-to-right, 12 right-to-wrong. This matches Astra's figures. The confirmation questions appear in no
  forward request before pass 27, and the four splits are disjoint (checked against `forward_calls.jsonl`).

### Other interventions tested (all on selection questions, all removed by Astra)
| pass | intervention | n | accuracy difference | wrong->right / right->wrong | margin difference (SE) |
|---|---|---|---|---|---|
| 2 | L26 unit 4131 output x 0.5 | 128 | 0.00 pp | 1 / 1 | +0.03 (0.06) |
| 3 | L26 unit 4131 output x 0 | 128 | +1.56 pp | 3 / 1 | -0.03 (0.12) |
| 5 | L31 units 4775, 6367, 6760, 13: remove projection on the "A" readout direction, strength 1 | 128 | 0.00 pp | 1 / 1 | +0.03 (0.08) |
| 6 | same, strength 3 | 128 | +0.78 pp | 4 / 3 | -0.05 (0.25) |
| 7 | same, strength 3 | 512 | -0.78 pp | 7 / 11 | -0.10 (0.12) |
| 9 | L25 head 9 output x 1.5 | 128 | +0.78 pp | 2 / 1 | -0.09 (0.05) |
| 10 | L25 head 9 output x 0.5 | 128 | 0.00 pp | 1 / 1 | -0.05 (0.05) |
| 11 | L30 head 24 output x 0.5 | 128 | 0.00 pp | 0 / 0 | +0.00 (0.01) |
| 14 | L31 unit 5937, variable part removed | 128 | -0.78 pp | 0 / 1 | +0.27 (0.08) |
| 15 | L26 unit 4131 output x -1 | 128 | +3.13 pp | 6 / 2 | -0.38 (0.27) |
| 16 | L26 unit 4131 output x -1 | 512 | +0.98 pp | 10 / 5 | -0.29 (0.14) |
| 17 | L26 unit 4131, only the letter-readout part x -1 | 128 | 0.00 pp | 1 / 1 | -0.43 (0.16) |
| 18 | L26 unit 4131, only the other part x -1 | 128 | +1.56 pp | 3 / 1 | +0.39 (0.20) |
| 19 | same as 18 | 512 | +0.98 pp | 7 / 2 | +0.19 (0.11) |
| 20 | L26 unit 4131, the other part x -3 | 128 | +3.13 pp | 5 / 1 | +0.31 (0.42) |
| 23 | L26 unit 4131, the other part x +5 (opposite direction) | 128 | +0.78 pp | 2 / 1 | -1.05 (0.34) |

The first 128 selection questions were part of 17 score passes (2 to 23), and the first 512 of passes 7, 16, 19 and 21+22.

### Diagnosis numbers (Astra's, from the unedited model's results on the 3,248 non-confirmation questions)
| letter | times correct answer | times predicted | recall |
|---|---|---|---|
| A | 804 | 575 | 49.0 % |
| B | 839 | 919 | 64.1 % |
| C | 826 | 843 | 62.8 % |
| D | 779 | 911 | 66.0 % |

## Observations
- `[OBSERVATION]` Astra submitted no edit. On the test split the submitted weights score exactly as the unedited
  model (0.5719, n = 1,273).
- `[OBSERVATION]` The unedited model predicts A less often than A is correct (575 vs 804 on 3,248 practice
  questions; recall for A 49.0 % against 62.8 to 66.0 % for B, C, D). The same shape is in the test split's
  predictions (A 252 of 1,273).
- `[OBSERVATION]` The locked candidate gained +1.95 pp (SE 0.73) on the 512 questions used to choose it and
  +0.68 pp (SE 0.54) on 1,024 untouched questions; on 256 other untouched questions it lost 0.78 pp (SE 0.78).
- `[OBSERVATION]` The gain on the first 128 selection questions, which were part of 17 score passes, was
  larger than on later questions for the same edit in two cases: unit 4131 x -1 gave +3.13 pp on 128 and
  +0.98 pp on 512 (passes 15, 16).
- `[OBSERVATION]` Edits that moved predictions toward A (passes 5 to 7) changed the letter distribution in the
  intended direction (A predictions 26 -> 33 of 128 at strength 3) without an accuracy gain; on 512 questions the
  difference was -0.78 pp (SE 0.83).
- `[OBSERVATION]` Scaling single attention heads' outputs (L25 head 9, L30 head 24) by 0.5 or 1.5 changed at most
  3 answers of 128.
- `[OBSERVATION]` Astra's descriptive decomposition on the confirmation set (`final_results.json`,
  `letter_shift_diagnostic`): adding the constant per-letter shift estimated on development questions to the
  original model's scores gives +0.29 pp, and the edited model gains 0.00 pp over that. Astra labels this
  "descriptive score decomposition, not additive causal attribution".

- `[OBSERVATION]` (Claude's re-analysis of `score_27..30.jsonl`, not Astra's) On the 1,024 confirmation questions
  the locked candidate moved the gold-answer margin in the direction it already had: +3.81 on average on the 588
  questions with a positive margin before the edit, -4.39 on the 436 with a negative one; correlation between the
  margin before and its change 0.72; mean absolute margin 14.11 -> 18.25. The same holds on the selection set
  (+3.63 / -4.54, correlation 0.74) and the reserve set (+3.58 / -4.31, 0.74). 31 of 1,024 answers changed
  correctness.

## Negative / null results
- No intervention tested in 30 passes produced an accuracy gain that held on untouched questions at Astra's
  pre-set threshold.
- Correcting the answer-letter imbalance directly (passes 5 to 7, 17) did not raise accuracy.
- The experiment's primary outcome is null: test-split accuracy difference 0.0, because nothing was submitted.

## Failed / excluded runs
- Two attempts at pass 21 returned nothing and were not counted (see Deviations).
- One Codex reply at reply 41 came back empty and was retried (`raw/run1/turns/t041_fail1`).
- One proposed edit (a mean-preserving edit in layer 27) was not installed: Astra's own precondition check failed
  before `save_edit`.

## Anomalies
- The unedited model's predictions on the same 512 selection questions differ on 2 questions between pass 19
  (before the restart, one request of 512) and passes 21+22 (after it, two requests of 256): `train-02263`
  (B -> D, an exact tie between B and D before) and `dev-00258` (A -> B, margin +0.06 -> -0.06). Original accuracy
  on those 512 is 56.84 % before and 56.64 % after. Each paired comparison uses the baseline from its own pass.
  Float16 scoring with different batch padding is the likely cause; it was not investigated further.
- Arrays `fwd_1`, `fwd_4`, `fwd_8`, `fwd_12`, `fwd_13` hold the post-norm convention for layer 32; `fwd_24`,
  `fwd_25` hold the pre-norm convention. They are not comparable at layer 32.

## Scope
One session (`run1`), one editor model at effort `xhigh`, one edited model, 30 forward passes, chat-prompt letter
scoring only, MedQA USMLE 4-option. Edits were restricted to weights other than the embeddings, the output layer
and the final norm, with no gradients and no simulation of the network in the kernel. The locked candidate was
never scored on the test split (only the submitted, unedited weights were).

## What this experiment does NOT establish
- That failure-guided editing cannot improve Phi-3.5 on MedQA. One session with 30 passes found nothing that met
  its own threshold; other budgets, effort levels, prompts or sessions were not run.
- That the L26 unit 4131 edit has no effect. Its confirmation estimate is +0.68 pp with an approximate 95 %
  interval of -0.38 to +1.75 pp; the experiment could not distinguish a small gain from none.
- Anything about medical knowledge in the weights. Astra reports that no improvement in the use of medical
  content was established, and that letter preference remained a confound for the candidate's gain.
- Behaviour under the bare prompt of EXP-04 Protocol C, or with generated reasoning. Neither was measured.

## Reproducibility
- commit: passes 21 to 30 ran from the working tree committed as `3f24402` and `addec46` (2026-11-22 21:49 and
  21:55). Passes 1 to 20 ran from the working tree of 16:39 that day, before commit `fc725bc`; that is not an
  exact committed state.
- seed: Astra's split seed 20261122 (`work/notes.md`). Scoring is deterministic up to float16 ties (see Anomalies).
- environment: Apple Silicon Mac, MPS, float16, transformers-native Phi-3; Codex CLI 0.169 (ChatGPT.app binary).
- command: `python experiments/exp_06_failure_guided_editing/run.py --name run1 --budget 30 --effort xhigh`
  (rerunning resumes from `raw/run1/state.json`; a new `--name` starts a new session).

## Status
Awaiting researcher review. Not promoted to `research/07_RESULTS_GROUND_TRUTH.md`.
