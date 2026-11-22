---
type: experiment
id: EXP-01
status: done
tags: [experiment, mit-replication, astra]
---

# EXP-01 - MIT exp 3: reading a language model's middle blocks

## Question
Given only the four transformer blocks of a small character-level language model (no embeddings, no output
layer), can Astra say which corpus it was trained on and which heads copy earlier text? This is experiment 3 of
[[Takehi et al. 2026 - Weight-Space Programming]].

## Setup
- Reader: `gpt-6-astra` via `codex exec`, effort `xhigh`, through the [[Astra harness]].
- Networks: MIT's released `blocks.npz` for python / romaji / english (4 layers x 6 heads, width 384).
- Astra may run Python on the weights (up to 100 runs, 600 s each) but no forward pass and no search.
- Two conversations per network.

## Deviations from the source protocol
macOS sandbox instead of bubblewrap; a few Codex tool definitions cannot be removed, so every turn is audited
for tool use instead (all clean); ground truth is MIT's, not re-measured. Full list in
`experiments/exp_01_mit3_language/RESULTS.md`.

## Runs
- 2026-11-21: `python experiments/exp_01_mit3_language/run.py --parallel 3` -> `experiments/exp_01_mit3_language/raw/runs/`

## Results (factual)
| Network | Corpus guess (P), run 0 / run 1 | Copy heads named | True copy heads |
|---|---|---|---|
| python | python (0.65) / python (0.70) | 2.1, 3.0, 3.1, 3.5 (both runs) | 2.1, 3.1, 3.5 |
| romaji | romaji (0.58) / romaji (0.60) | 3.3 (both runs) | 3.3 |
| english | english (0.55) / english (0.55) | 3.1 (both runs) | 3.1 |

- Corpus right in 6 of 6 conversations (MIT: 6 of 6).
- All 10 true copying heads found; one extra head (python 3.0, measured induction score 0.018) is labelled
  `copy` in both Python conversations. MIT's runs had no false positives.
- 6 to 10 code runs per conversation (MIT: 6 to 11).

## What this does and does not show
Shows that the reading result reproduces with our own Astra access on MIT's networks. Does not re-verify the
ground truth, and two conversations per network are not a rate estimate.

## Visualizations
See [[Visualization index]] (3D view of the python network's blocks with the copying heads marked - planned).

## Links
[[MIT replication spec (exp 3, 7, 8)]] - [[Weight-space programming]] - [[2026-11-21 Session 01]]
