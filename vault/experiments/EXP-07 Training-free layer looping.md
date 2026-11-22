---
type: experiment
id: EXP-07
status: running
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
- queued 2026-11-22: `--model phi --split dev --limit 600 --sweep defaults position` -> `processed/scan_phi_dev_600.jsonl`

## Results (factual)
Pending.

## Links
[[Layer looping]] - [[EXP-04 MedQA baseline - Phi-3.5 vs MediPhi]]
