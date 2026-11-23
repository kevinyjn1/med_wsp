---
type: experiment
id: EXP-07
status: first scan done, awaiting review
tags: [experiment, layer-looping, medqa]
---

# EXP-07 - Training-free layer looping

## Question
Filippos's "Experiment 2": does re-running selected transformer layers at inference improve MedQA accuracy
without fine-tuning, and which block should be repeated? Methods: [[Chen et al. 2026 - Training-Free Looped Transformers]]
and [[Lys et al. 2026 - Inner Loop Inference]]; summary in [[Layer looping]].

## Setup
- Code: `wsp/looping.py` (rules `naive`, `chen`, `lys`), tests in `tests/test_looping.py`,
  runner `experiments/exp_07_layer_looping/run.py`.
- Scoring: single-pass letter scoring (no key/value cache, so no generation-time cache handling is needed).
- Every configuration is compared with the unmodified model on the same questions: accuracy difference,
  questions fixed and broken, McNemar p, mean margin change with its standard error.
- Planned sweeps: `defaults` (4-layer window at layers 14-17: Chen K=2 and K=3, Lys uniform R=3, naive K=2),
  `position` (4-layer windows at starts 2, 4, ..., 26 with Chen K=2 and naive K=2), then `around` the best start.
- Selection on MedQA dev (first 600 questions), confirmation on the test split.

## What to expect (from the papers)
Gains of about 1 point or less on multiple-choice benchmarks (Chen: +0.30 to +2.64 pp with per-task selected
settings; Lys: -0.55 to +0.98 pp). Detecting 1 pp with a paired test needs on the order of 4,000 questions, so
the 600-question scan can only rank windows by the margin, not confirm an accuracy gain.

## Runs
- 2026-11-23 02:20 to 09:07: `--model phi --split dev --limit 600 --sweep defaults position` ->
  `processed/scan_phi_dev_600.jsonl`. 28 configurations. It took 6 h 47 min instead of about 1 h 50 min because
  the Mac slept from 03:32 to 09:02.
- Not run yet: the `around` sweep and the confirmation on the test split.

## Results (factual, awaiting review)
Handoff: `experiments/exp_07_layer_looping/RESULTS.md` (full table). Unmodified Phi-3.5: 56.67 % of 600, margin +3.78.
- `[OBSERVATION]` No configuration raises accuracy with p < 0.05. 10 of 28 are above the unmodified model, 5 equal,
  13 below.
- `[OBSERVATION]` Largest gain: damped rule (Chen, 2 sub-steps) on layers 10-13, +1.83 points (35 fixed, 24
  broken, p = 0.19). It is also the only configuration whose mean margin rises by more than 2 standard errors
  (+0.634, SE 0.189).
- `[OBSERVATION]` Plain repetition of early layers hurts: layers 2-5 -5.17 points (p = 0.005), 4-7 -7.00
  (p < 0.001), 6-9 -3.83 (p = 0.023).
- `[OBSERVATION]` The deeper the window, the fewer answers change: plain repetition changes 117 to 120 of 600
  answers at layers 2-5 and 4-7 and 7 at layers 26-29; the damped rule changes 1 to 5 from layer 22 on.
- `[OBSERVATION]` The papers' default recipes on layers 14-17: -0.33 to +0.33 points.
- Not established: that looping cannot help (600 questions detect only differences of about 4 points or more);
  that layers 10-13 are a real gain (best of 28 on the same questions, not confirmed elsewhere).

## Visualization
Viewer dataset `looping_phi_dev_600` ([[Visualization index]]).

![[viz_exp07_looping_scan.jpg]]

## Links
[[Layer looping]] - [[EXP-04 MedQA baseline - Phi-3.5 vs MediPhi]]
