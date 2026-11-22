# EXP-04 Results - MedQA: Phi-3.5-mini-instruct vs MediPhi vs MediPhi-Instruct

## Experiment
First step of Filippos Bellos's plan (2026-11-21): "evaluating Phi-3.5 and MediPhi on MedQA, focusing on questions
where MediPhi succeeds but Phi fails". No experiment contract was written beforehand. The scoring protocol, the
choice of splits and the addition of MediPhi-Instruct are Claude's provisional choices, not researcher decisions.

## Run status
COMPLETE for the two protocols below.

## What actually ran
- Models (local Hugging Face snapshots, transformers-native Phi-3 code, float16 on Apple MPS):
  `microsoft/Phi-3.5-mini-instruct`, `microsoft/MediPhi`, `microsoft/MediPhi-Instruct`.
- Data: MedQA USMLE, 4 options, official splits from `bigbio/med_qa` (`med_qa_en_4options_source`):
  test 1,273, dev 1,272, and the first 3,000 of the 10,178 training questions.
- Protocol A, letter scoring (`run.py`, `wsp/medqa.py`): chat prompt = one instruction line, the question, options
  A-D, "Answer:"; one forward pass; the answer is the letter with the highest next-token probability, summing the
  spellings "A" and " A". Eager attention, batch 16. Run 2026-11-21 (Phi-3.5, MediPhi) and 2026-11-22 (MediPhi-Instruct).
- Protocol B, reasoning (`run_cot.py`): the prompt asks to think step by step and end with "Answer: X"; greedy
  decoding, at most 512 new tokens, default (non-eager) attention, batch 8; the last "Answer: X" is parsed.
  First 400 test questions. Run 2026-11-22.

## Deviations from contract
- No contract exists.
- A first reasoning run with eager attention was killed after it grew to 21 GB and stalled the machine; no
  result from it was kept.

## Artifacts
- `processed/{phi,mediphi,mediphi_instruct}_{test,dev,train}.jsonl` - per question: prediction, correctness,
  margin (log-prob of the gold letter minus the best other letter), letter log-probs
- `processed/cot_{phi,mediphi,mediphi_instruct}_test_400.jsonl` - per question: generated text, parsed answer
- `run.py`, `run_cot.py`

## Numerical results

### Protocol A - letter scoring (accuracy)
| Split | n | Phi-3.5 | MediPhi | MediPhi-Instruct |
|---|---|---|---|---|
| test | 1,273 | 0.5719 | 0.5632 | 0.5648 |
| dev | 1,272 | 0.5841 | 0.5951 | 0.5802 |
| train (first 3,000) | 3,000 | 0.6037 | 0.5923 | 0.5780 |

Standard error of one accuracy: about 0.014 at n = 1,273 and 0.009 at n = 3,000.

Paired comparison with Phi-3.5 (questions only one model answers correctly; exact two-sided sign test):
| Split | only Phi-3.5 / only MediPhi | p | only Phi-3.5 / only MediPhi-Instruct | p |
|---|---|---|---|---|
| test | 57 / 46 | 0.324 | 68 / 59 | 0.478 |
| dev | 39 / 53 | 0.175 | 78 / 73 | 0.745 |
| train (first 3,000) | 129 / 95 | 0.027 | 214 / 137 | < 0.001 |

- Same predicted letter as Phi-3.5: MediPhi about 90 % of questions, MediPhi-Instruct about 85 %.
- Mean gold-answer margin (test / dev / train): Phi-3.5 +3.89 / +4.54 / +5.12; MediPhi +2.81 / +3.25 / +3.43;
  MediPhi-Instruct +1.28 / +1.53 / +1.55.
- Share of next-token probability on the four letters: 0.999 or more for all three models.
- Phi-3.5 predicted letters on test: A 252, B 324, C 364, D 333; gold letters: A 353, B 309, C 346, D 265.

### Protocol B - reasoning, first 400 test questions
| Model | Accuracy | Parsed | Mean new tokens | Letter-scoring accuracy on the same 400 |
|---|---|---|---|---|
| Phi-3.5 | 0.6300 | 0.995 | 148 | 0.5825 |
| MediPhi | 0.6000 | 0.988 | 160 | 0.5775 |
| MediPhi-Instruct | 0.5775 | 1.000 | 6 | 0.5975 |

Standard error of one accuracy: about 0.024. Paired with Phi-3.5: MediPhi 42 / 30 (p = 0.195);
MediPhi-Instruct 50 / 29 (p = 0.024), counted as only Phi-3.5 right / only the other model right.

## Observations
- `[OBSERVATION]` Under letter scoring no medical model is more accurate than Phi-3.5 on any split by more than
  noise, and on the 3,000-question pool both are less accurate (MediPhi-Instruct clearly so).
- `[OBSERVATION]` MediPhi-Instruct did not reason when asked to: 97 % of its replies have at most 8 tokens
  ("Answer: X"). Its Protocol B number is therefore a direct-answer score, not a reasoning score.
- `[OBSERVATION]` With reasoning, Phi-3.5 gains 4.8 points over its own letter score on the same 400 questions;
  MediPhi gains 2.3.
- `[OBSERVATION]` The MediPhi paper's appendix Table 13 reports MedQA 0.486 for Phi-3.5, 0.491 for MediPhi and
  0.548 for MediPhi-Instruct, with no protocol given. Our Phi-3.5 numbers (0.57 to 0.63) are above all three.

## Negative / null results
- No MedQA improvement of MediPhi or MediPhi-Instruct over Phi-3.5 was found under either protocol.

## Failed / excluded runs
- The killed eager-attention reasoning run (see deviations).

## Anomalies
- The reasoning prompt is our own wording; a different prompt (for example a system prompt) might change whether
  MediPhi-Instruct reasons.

## Scope
One prompt per protocol, one decoding setting, MedQA USMLE 4-option English only; Protocol B on 400 questions.

## What this experiment does NOT establish
- That MediPhi is not better at medicine in general: the paper's headline gains are on clinical tasks (CLUE+), not MedQA.
- Why the paper's MedQA numbers differ from ours (protocol unknown).
- Anything about other prompts, few-shot settings, or sampled decoding.

## Reproducibility
- commands: `python experiments/exp_04_medqa_baseline/run.py`; `... run.py --models mediphi_instruct`;
  `python experiments/exp_04_medqa_baseline/run_cot.py --split test --limit 400`
- environment: Apple M4 Max, transformers 5.24.0, torch 2.15.0, float16 on MPS
- seeds: none needed (no sampling)

## Status
Awaiting researcher review.
