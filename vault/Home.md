---
type: home
tags: [moc]
---

# med_wsp - Weight-space programming for medical reasoning

Map of content for the project vault. The repo's `research/` folder holds the approved scientific state;
this vault is the working notebook: what was read, what was run, and what happened in each session.

## Project in one paragraph
Use an LLM that reads and writes weights (GPT-6 Astra, through the Codex CLI) together with standard
interpretability tools to find which layers, heads, MLPs and features make a medical fine-tune
(MediPhi) better than its base (Phi-3.5-mini-instruct) on MedQA, then try to move that improvement into
the base model with targeted weight edits and no fine-tuning. Two side directions: failure-guided weight
editing without a donor model, and training-free looping of selected layers.

## Sessions
- [[2026-11-21 Session 01]]

## Experiments
| ID | Note | What it is |
|---|---|---|
| EXP-01 | [[EXP-01 MIT exp 3 - reading a language model's middle blocks]] | MIT replication: which corpus, which heads copy |
| EXP-02 | [[EXP-02 MIT exp 7 - which model is sycophantic]] | MIT replication: honest vs sycophant Gemma 3 270M |
| EXP-03 | [[EXP-03 MIT exp 8 - editing Qwen3.5-0.8B for math]] | MIT replication: loop-direction edit |
| EXP-04 | [[EXP-04 MedQA baseline - Phi-3.5 vs MediPhi]] | Accuracy, contrast sets |
| EXP-05 | [[EXP-05 Medical circuit discovery and transfer]] | Weight diff, patching, dictionaries, Astra edits |
| EXP-06 | [[EXP-06 Failure-guided weight editing]] | No donor model |
| EXP-07 | [[EXP-07 Training-free layer looping]] | Repeat selected layers at inference |

Replication protocol details: [[MIT replication spec (exp 3, 7, 8)]]

## Papers
- [[Takehi et al. 2026 - Weight-Space Programming]] (the MIT CSAIL blog post we replicate)
- [[Li et al. 2026 - WeightCode]]
- [[Tan et al. 2026 - Efficient Transcoder Adaptation]] (closest prior art)
- [[Lindsey et al. 2024 - Sparse Crosscoders]]
- [[Ameisen et al. 2025 - Circuit Tracing]]
- [[Chen et al. 2026 - Training-Free Looped Transformers]]
- [[Lys et al. 2026 - Inner Loop Inference]]
- [[Corbeil et al. 2025 - MediPhi]]

## Concepts
- [[Weight-space programming]]
- [[Transcoders]] - [[Crosscoders]] - [[Attribution graphs]] - [[Activation patching]]
- [[Layer looping]]
- [[Astra harness]] (how we run GPT-6 Astra through Codex)

## Visualizations
- [[Visualization index]]

## Where things live
| Thing | Path (repo root = `med_wsp/`) |
|---|---|
| Code shared by experiments | `wsp/` |
| Astra harness (Codex driver, sandbox, loop) | `wsp/harness/` |
| Experiments (code, raw logs, processed results) | `experiments/exp_XX_*/` |
| Approved scientific state | `research/` |
| Session checkpoint for resuming | `docs/CURRENT_STATE.md` |
| Paper PDFs and text (local only, not in git) | `vault/papers/pdf/` |
| MIT logs repo (local only) | `external/lm_understand-logs/` |
