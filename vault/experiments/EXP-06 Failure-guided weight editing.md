---
type: experiment
id: EXP-06
status: running
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

## Runs
- `run1`, started 2026-11-22 16:39, budget 30: `experiments/exp_06_failure_guided_editing/raw/run1/`

## Results (factual)
Pending. First turns of `run1`: Astra set aside 1,024 practice questions for confirmation, then found that the
model predicts A on 575 questions where A is correct on 804 (recall 49.0 % vs 62.8-66.0 % for B, C, D), and
started testing late-layer MLP neurons for a constant letter preference.

## Links
[[Astra harness]] - [[Weight-space programming]] - [[EXP-04 MedQA baseline - Phi-3.5 vs MediPhi]]
