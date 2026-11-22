---
type: experiment
id: EXP-04
status: done
tags: [experiment, medqa, baseline]
---

# EXP-04 - MedQA baseline: Phi-3.5 vs MediPhi (and MediPhi-Instruct)

## Question
Filippos's first step: evaluate Phi-3.5 and MediPhi on MedQA and find the questions MediPhi gets right and Phi
gets wrong. Before that set means anything: is MediPhi actually better on MedQA?

## Setup
- MedQA USMLE 4 options from `bigbio/med_qa`: test 1,273, dev 1,272, first 3,000 training questions.
- Protocol A (letter scoring): one forward pass, pick the letter with the highest next-token probability.
- Protocol B (reasoning): "think step by step ... Answer: X", greedy, 512 new tokens, first 400 test questions.
- MediPhi-Instruct was added because the MediPhi paper reports its MedQA gain there, not for `microsoft/MediPhi`.

## Results (factual)
| | Phi-3.5 | MediPhi | MediPhi-Instruct |
|---|---|---|---|
| Letter scoring, test (1,273) | 0.572 | 0.563 | 0.565 |
| Letter scoring, dev (1,272) | 0.584 | 0.595 | 0.580 |
| Letter scoring, train pool (3,000) | 0.604 | 0.592 | 0.578 |
| Reasoning, 400 test questions | 0.630 | 0.600 | 0.578 (answers directly, 6 tokens on average) |

- No medical model beats Phi-3.5 on MedQA under either protocol. On the 3,000-question pool both are slightly
  worse (sign test p = 0.027 and p < 0.001).
- Disagreement is roughly symmetric: on test, 46 questions only MediPhi gets right and 57 only Phi-3.5 gets right.
- MediPhi-Instruct ignores the request to reason (97 % of replies are just "Answer: X").
- The paper's appendix reports 0.486 / 0.491 / 0.548 with no protocol; our Phi-3.5 numbers are above all three.

Full tables, paired tests and caveats: `experiments/exp_04_medqa_baseline/RESULTS.md`.

## What this does and does not show
Shows that, with these two protocols, MedQA has no MediPhi advantage to explain. Does not show that MediPhi is
not better elsewhere (its paper's gains are on clinical tasks), and does not explain the paper's MedQA numbers.

## Consequence for the plan (open, researcher's call)
The "MediPhi right, Phi wrong" questions exist (46 on test, 53 on dev, 95 in the pool) but are matched by as
many in the other direction, so they look like disagreement between two similar models rather than a medical
gain. Options to put to Filippos: a benchmark where the fine-tune clearly wins (the paper's clinical tasks, for
example ICD-10 coding), another pair (Gemma vs MedGemma, gated for now), or leaning on the donor-free experiments
([[EXP-06 Failure-guided weight editing]], [[EXP-07 Training-free layer looping]]).

## Links
[[Corbeil et al. 2025 - MediPhi]] - [[EXP-05 Medical circuit discovery and transfer]] - [[2026-11-21 Session 01]]
