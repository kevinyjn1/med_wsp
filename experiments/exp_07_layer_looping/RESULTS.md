# EXP-07 Results - Training-free layer looping on Phi-3.5, first scan (MedQA dev, 600 questions)

## Experiment
Filippos Bellos's "Experiment 2" (2026-11-21): re-run selected transformer layers at inference, with no
fine-tuning, and see whether MedQA accuracy improves and which layers to repeat. No formal experiment contract was
written. The working specification is `vault/experiments/EXP-07 Training-free layer looping.md` and the docstring
of `run.py`. Window length, step counts, the sweep grid and the 600-question limit are Claude's provisional
choices, not researcher decisions.

## Run status
PARTIAL. The two planned first sweeps (`defaults`, `position`) are complete. The planned follow-up sweep around
the best window (`around`) and the confirmation on the test split were not run.

## What actually ran
- Model: `microsoft/Phi-3.5-mini-instruct`, float16 on Apple MPS, 32 layers. No weights changed.
- Questions: the first 600 of the MedQA dev split. Prompt: the chat prompt of EXP-04 Protocol A (instruction line,
  question, options, "Answer:"). Score: next-token log-probabilities of the four letters in one forward pass
  without a key/value cache. Margin = log-probability of the correct letter minus the best wrong letter.
- Looping (`wsp/looping.py`): a window of 4 consecutive layers g is applied to its input x0 with one of three
  rules.
  - Plain repetition: x <- g(x), twice in total.
  - Damped (Chen et al. 2026): anchor = g(x0); then K sub-steps x <- x + (g(x) - x) / K, the first reusing the
    anchor; the anchor is not mixed back in (beta = 0).
  - Averaged (Lys et al. 2026, "uniform"): h(t) = g(mean of x0 and h(1..t-1)), R = 3 passes.
- Configurations: 28 distinct ones.
  - `defaults`: layers 14-17 with damped K=2, damped K=3, averaged R=3, plain repetition x2.
  - `position`: windows starting at layers 2, 4, ..., 26 with damped K=2 and plain repetition x2 (26 runs, two of
    them the same as in `defaults` and read from the saved files).
- Every configuration is compared with the unmodified model on the same 600 questions: accuracy difference,
  questions fixed (wrong to right) and broken (right to wrong), exact two-sided sign test on fixed vs broken
  (stored as `mcnemar_p`), mean margin change with its standard error.
- Ran 2026-11-23 02:20 to 09:07 from `scripts/queue_sweeps.sh`.

## Deviations from contract
- No contract exists.
- From the working specification: the `around` sweep and the test-split confirmation were not run.

## Artifacts
- `processed/scan_phi_dev_600.jsonl` - one line per run (31 lines: baseline, 28 configurations, 2 repeats).
- `processed/per_question_phi_dev_600/<config>.jsonl` - per-question prediction, correctness and margin.
- Log (not committed): `logs/exp07_scan.log`.
- Viewer dataset `looping_phi_dev_600` (`python -m wsp.viz_export looping`), screenshot
  `vault/attachments/viz_exp07_looping_scan.jpg`.

## Numerical results
Unmodified model: accuracy 0.5667 (340 of 600), mean margin +3.777.

| Repeated layers | Rule | Steps | Accuracy | Change (points) | Fixed / broken | Sign test p | Margin change (SE) |
|---|---|---|---|---|---|---|---|
| 2-5 | damped (Chen) | 2 | 0.5533 | -1.33 | 27 / 35 | 0.374 | -0.639 (0.267) |
| 2-5 | plain repetition | 2 | 0.5150 | -5.17 | 43 / 74 | 0.005 | -2.666 (0.419) |
| 4-7 | damped (Chen) | 2 | 0.5500 | -1.67 | 38 / 48 | 0.332 | -0.417 (0.290) |
| 4-7 | plain repetition | 2 | 0.4967 | -7.00 | 39 / 81 | < 0.001 | -2.394 (0.428) |
| 6-9 | damped (Chen) | 2 | 0.5550 | -1.17 | 32 / 39 | 0.477 | -0.409 (0.264) |
| 6-9 | plain repetition | 2 | 0.5283 | -3.83 | 36 / 59 | 0.023 | -2.950 (0.428) |
| 8-11 | damped (Chen) | 2 | 0.5517 | -1.50 | 28 / 37 | 0.321 | -0.448 (0.231) |
| 8-11 | plain repetition | 2 | 0.5567 | -1.00 | 47 / 53 | 0.617 | -1.456 (0.342) |
| 10-13 | damped (Chen) | 2 | 0.5850 | +1.83 | 35 / 24 | 0.193 | +0.634 (0.189) |
| 10-13 | plain repetition | 2 | 0.5733 | +0.67 | 38 / 34 | 0.724 | -0.263 (0.283) |
| 12-15 | damped (Chen) | 2 | 0.5767 | +1.00 | 20 / 14 | 0.392 | +0.024 (0.161) |
| 12-15 | plain repetition | 2 | 0.5667 | +0.00 | 28 / 28 | 1.000 | -0.616 (0.294) |
| 14-17 | damped (Chen) | 2 | 0.5683 | +0.17 | 12 / 11 | 1.000 | -0.146 (0.156) |
| 14-17 | damped (Chen) | 3 | 0.5700 | +0.33 | 18 / 16 | 0.864 | -0.321 (0.197) |
| 14-17 | averaged (Lys, uniform) | 3 | 0.5683 | +0.17 | 16 / 15 | 1.000 | -0.235 (0.175) |
| 14-17 | plain repetition | 2 | 0.5633 | -0.33 | 20 / 22 | 0.878 | -0.522 (0.213) |
| 16-19 | damped (Chen) | 2 | 0.5617 | -0.50 | 11 / 14 | 0.690 | -0.272 (0.093) |
| 16-19 | plain repetition | 2 | 0.5650 | -0.17 | 16 / 17 | 1.000 | -0.496 (0.149) |
| 18-21 | damped (Chen) | 2 | 0.5567 | -1.00 | 5 / 11 | 0.210 | -0.409 (0.104) |
| 18-21 | plain repetition | 2 | 0.5667 | +0.00 | 9 / 9 | 1.000 | -0.480 (0.100) |
| 20-23 | damped (Chen) | 2 | 0.5683 | +0.17 | 8 / 7 | 1.000 | -0.324 (0.098) |
| 20-23 | plain repetition | 2 | 0.5683 | +0.17 | 17 / 16 | 1.000 | -0.872 (0.221) |
| 22-25 | damped (Chen) | 2 | 0.5667 | +0.00 | 2 / 2 | 1.000 | -0.026 (0.043) |
| 22-25 | plain repetition | 2 | 0.5667 | +0.00 | 12 / 12 | 1.000 | -1.481 (0.303) |
| 24-27 | damped (Chen) | 2 | 0.5683 | +0.17 | 1 / 0 | 1.000 | -0.051 (0.052) |
| 24-27 | plain repetition | 2 | 0.5667 | +0.00 | 14 / 14 | 1.000 | -1.426 (0.255) |
| 26-29 | damped (Chen) | 2 | 0.5750 | +0.83 | 5 / 0 | 0.062 | -0.046 (0.034) |
| 26-29 | plain repetition | 2 | 0.5650 | -0.17 | 3 / 4 | 1.000 | -1.666 (0.296) |

Counts over the 28 configurations: accuracy above the unmodified model in 10, equal in 5, below in 13. The
p-values are not corrected for the 28 comparisons.

## Observations
- `[OBSERVATION]` No configuration raises accuracy with p < 0.05. The largest gain is the damped rule on layers
  10-13: +1.83 points (35 fixed, 24 broken, p = 0.193).
- `[OBSERVATION]` The same configuration is the only one whose mean margin rises by more than 2 standard errors:
  +0.634 (SE 0.189). The mean margin rises in 2 of 28 configurations and falls in 26.
- `[OBSERVATION]` Plain repetition of an early window lowers accuracy: layers 2-5 by 5.17 points (p = 0.005),
  layers 4-7 by 7.00 points (p < 0.001), layers 6-9 by 3.83 points (p = 0.023). These are the only three
  configurations with p < 0.05.
- `[OBSERVATION]` The number of answers that change falls with depth. Plain repetition changes 117 to 120 of 600
  answers for windows starting at layers 2 and 4, and 7 for layers 26-29. The damped rule changes 62 to 86 answers
  for windows starting at layers 2 to 8, and 1 to 5 for windows starting at layers 22 to 26.
- `[OBSERVATION]` At every window the mean margin under the damped rule is higher than under plain repetition.
- `[OBSERVATION]` The papers' default recipes on layers 14-17 change accuracy by -0.33 to +0.33 points (12 to 20
  questions fixed, 11 to 22 broken).
- `[OBSERVATION]` Damped rule on layers 26-29: 5 questions fixed and none broken (p = 0.0625, the smallest value
  a 5-to-0 split can give).

## Negative / null results
- None of the 28 looping configurations improves accuracy on 600 dev questions at p < 0.05, corrected or not.
- All four default recipes from the two papers are within 0.33 points of the unmodified model.

## Failed / excluded runs
- None in this run. The first launch of the sweep queue (2026-11-22 23:31) failed before loading a model (shell
  quoting error, exit 127) and wrote nothing to `processed/`.

## Anomalies
- Run time. The first 19 runs took about 200 to 330 s each. From the window starting at layer 16 on, runs took
  508 to 4,331 s (wall time), except the last one (189 s), which ran after the Mac woke. The Mac entered idle sleep at 03:32 and stayed in sleep and short maintenance wakes
  until 09:02 (`pmset -g log`), so the process ran only in bursts. The stored `seconds` values of those runs are
  wall time and do not measure the cost of looping.
- No sign that the pauses changed any result was found, and none was looked for beyond the fact that every run
  wrote 600 per-question records. No run was repeated to check.
- Because the run held a GPU lock from 02:20 to 09:07, the Astra donor session (EXP-05c) waited for its sixth
  forward pass from 02:33 to 09:07.

## Scope
- One model (Phi-3.5), one prompt (chat), 600 dev questions, single-pass letter scoring.
- Window length 4 only. Two rules across positions, each with 2 steps. Three and more steps and the averaged rule
  only at layers 14-17. Windows starting at odd layers, at layer 0 and at layers 27 or 28 were not run.
- No generated answers, no reasoning-style prompt, no bare prompt, no MediPhi model.

## What this experiment does NOT establish
- It does not establish that looping cannot help. With 60 to 120 answers changing, the standard error of the
  accuracy difference on 600 questions is 1.3 to 1.8 points, so only differences of roughly 4 points or more are
  reliably detectable; the papers report gains of about 1 point or less on multiple-choice tasks.
- It does not establish that the damped rule on layers 10-13 helps. That window was picked as the best of 28 on
  the same questions, its accuracy gain has p = 0.193, and it has not been scored on any other question.
- It says nothing about other window lengths, more steps, or other models.
- It does not test the papers' own evaluation settings (generation, their benchmarks, their models).

## Reproducibility
- Commit at run time: `98112a8` (`run.py` and `wsp/looping.py` last changed in `6310308`).
- Command: `python -m wsp.gpu_lock --name batch -- python experiments/exp_07_layer_looping/run.py --model phi --split dev --limit 600 --sweep defaults position`
- No randomness. Batch size 16.
- The run resumes from `processed/scan_phi_dev_600.jsonl`, so a re-run with the file present does nothing.
- Tests of the looping rules: `tests/test_looping.py`.

## Status
Awaiting researcher review. Not promoted to `research/07_RESULTS_GROUND_TRUTH.md`.
