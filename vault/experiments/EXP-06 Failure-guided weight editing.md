---
type: experiment
id: EXP-06
status: run1 complete, awaiting review
tags: [experiment, astra, weight-edit, medqa]
---

# EXP-06 - Failure-guided weight editing (no donor)

## Question
Filippos's "Experiment 1": give Astra a general model's wrong MedQA answers plus interpretability tools and
ask it to diagnose and repair the responsible weights directly, with no specialized donor model and no
fine-tuning. Does the edit improve held-out MedQA accuracy?

## Setup
- Model edited: Phi-3.5-mini-instruct. Editor: `gpt-6-astra` via `codex exec`, effort `xhigh`.
- Protocol: modeled on MIT experiment 8 ([[MIT replication spec (exp 3, 7, 8)]], section 3), rewritten for
  single-pass multiple-choice scoring. Code: `wsp/lab.py`, `wsp/lab_session.py`,
  `experiments/exp_06_failure_guided_editing/run.py`. The full task message is `raw/<run>/message.txt`.
- What Astra has:
  1. a persistent sandboxed Python kernel over the weights (`load`, `load_original`, `save_edit`, ...), CPU only;
  2. 4,272 practice questions (MedQA dev + first 3,000 training questions) with answers and the original
     model's result on each (so the failures are known);
  3. a budget of forward passes. A pass is either a score pass (up to 1,024 practice questions with original,
     edited or both weights, paired statistics) or a teacher-forced read (hidden states, MLP activations,
     attention, log-probs).
- Refused edits: final norm, input embedding, output layer, pure rescaling of a tensor. No gradients, no
  running the model in the kernel.
- Every reply must explain its step; conversations are restarted with the notebook every 40 replies.
- Final measurement: the submitted edit vs the unedited model on the MedQA test split (1,273 questions,
  never shown to Astra), paired.

## Deviations from the MIT exp 8 protocol
- Task and scoring: one forward pass per question, no generation, so a "generation pass" became a "score pass".
- Effort `xhigh` (MIT used `max`). Budget 30 passes (MIT: 10, 25, 50).
- Explanations required from the first turn (MIT added this mid-run).
- A fix of answer-letter bias is declared in scope but must be reported separately from content-related gains.

## Known tool caveat in `run1`
The task message says "hidden_states layer l is the residual stream after layer l". For l = 32 that was wrong in
`run1`: the lab returned what transformers reports as the last hidden state, which is after the final RMSNorm
(pinned by `tests/test_hidden_state_convention.py`). Astra raised the possibility itself at turn 16 and set out to
verify it. The lab was fixed on 2026-11-22 17:15 (layer 32 is now the pre-norm residual, checked in
`tests/integration_lab.py`), but `run1` was already running with the old behaviour and keeps it for the whole
session, so its arrays are self-consistent. Later sessions use the fixed tool.

## Interruption on 2026-11-22
The lab loaded the model inside the session's own process for every forward pass; on Apple MPS that memory was
not returned, and by pass 21 the process held 18 GB and the machine was swapping (pass 21 ran for two hours
without finishing). The process was stopped at 20 of 30 passes and the machine was restarted. Fix: every pass now
runs in a worker process (`wsp/lab_worker.py`). `run1` was resumed at 21:45 from its saved state with a note to
Astra: kernel variables lost, pass 21 not counted, and hidden_states layer 32 now means the pre-norm residual.

## Runs
- `run1`, started 2026-11-22 16:39, budget 30: `experiments/exp_06_failure_guided_editing/raw/run1/`

## Results (factual)
Full handoff: `experiments/exp_06_failure_guided_editing/RESULTS.md`. Astra's notebook stays on this machine at
`raw/run1/work/notes.md` (the work folder is not in git).

`run1` finished on 2026-11-22 23:06 after 69 replies, 36 kernel runs and all 30 forward passes.
**Astra submitted no edit.** Test split (1,273 questions): unedited 0.5719, submitted 0.5719, no tensor changed.

What it did, in order:
1. Split the 4,272 practice questions at random into discovery 1,200, selection 1,024, reserve 1,024 and final
   confirmation 1,024, and wrote down a retention rule before touching the confirmation set: keep an edit only if
   its gain there exceeds 2 standard errors and the sign test gives p < 0.05.
2. Diagnosis: the model predicts A on 575 of 3,248 questions where A is correct on 804 (recall 49.0 % vs
   62.8 to 66.0 % for B, C, D).
3. Tried to correct the letter preference directly (remove the "A" readout direction from four layer-31 MLP
   units): A predictions rose, accuracy did not (-0.78 pp, SE 0.83, on 512 questions).
4. Tried scaling single attention heads (L25 head 9, L30 head 24): at most 3 answers of 128 changed.
5. Found one MLP unit, layer 26 unit 4131, whose output always pushes against A. Reversing the part of its output
   that does not point along the letter readout, at strength -3, gave +1.95 pp (SE 0.73, 12 corrections vs 2
   regressions) on the 512 selection questions. Astra locked this candidate.
6. Confirmation on 1,024 untouched questions: +0.68 pp (SE 0.54), 19 corrections vs 12 regressions, p = 0.281.
   On 256 other untouched questions: -0.78 pp (SE 0.78). The rule failed, so it restored the original weights.

| question set | n | original | edited | difference | SE | sign test p |
|---|---|---|---|---|---|---|
| selection (used to choose the edit) | 512 | 56.64 % | 58.59 % | +1.95 pp | 0.73 | 0.013 |
| reserve (independent) | 256 | 60.94 % | 60.16 % | -0.78 pp | 0.78 | 0.625 |
| final confirmation (independent) | 1,024 | 57.52 % | 58.20 % | +0.68 pp | 0.54 | 0.281 |

I re-counted the confirmation result from the lab's per-question files (589 and 596 correct of 1,024) and checked
that the confirmation questions appear in no forward request before pass 27.

`[OBSERVATION]` The gain shrank from the questions used for selection to untouched ones (+1.95 pp to +0.68 pp).
`[OBSERVATION]` Astra's own reading: letter preference stayed a confound, and no improvement in the use of medical
content was established.

`[OBSERVATION]` (my re-analysis of the per-question files) The candidate mostly sharpened answers the model already
gave. On the 1,024 confirmation questions the margin of the correct answer moved by +3.81 on average where the
model was right and by -4.39 where it was wrong (correlation with the margin before: 0.72). Only 31 answers
changed correctness (19 fixed, 12 broken).

## Visualization
Viewer dataset `exp06_edit_run1` (`python -m wsp.viz_export exp06`).

![[viz_exp06_run1_unit.jpg]]
Top: which of layer 26's 8,192 MLP units was edited, and each unit's direct push on answer A (unit 4131 is the one
strong cell: -11.3, next largest 0.5). Middle: the unit's output direction before, the part kept, the part
multiplied by -3, and after.

![[viz_exp06_run1_per_question.jpg]]
One cell per practice question, sorted from most wrong (top left) to most right (bottom right) before the edit.
Middle column: red = fixed by the edit, blue = broken. Right column: change of the correct answer's margin.

## Not established
- That failure-guided editing cannot work: this is one session with 30 passes.
- That the unit-4131 edit does nothing: the confirmation interval is about -0.4 to +1.7 pp.
- The locked candidate was never scored on the test split. Doing so is a choice for the researcher (it spends
  part of the held-out set on a candidate that failed its own rule).

## Open questions for the researcher
- Run a second session with a larger budget, or wait for the donor session (EXP-05c) and compare?
- Should a later session target the bare prompt, where the model spreads its answer over non-letter tokens, in
  place of the chat prompt used here?

## Links
[[Astra harness]] - [[Weight-space programming]] - [[EXP-04 MedQA baseline - Phi-3.5 vs MediPhi]]
