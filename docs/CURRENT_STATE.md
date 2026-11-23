# Current State

Checkpoint written 2026-11-22 23:45 (session 01, still in progress). Detailed log: `vault/sessions/2026-11-21 Session 01.md`.

## Objective

1. Replicate MIT weight-space-programming experiments 3, 7 and 8 with GPT-6 Astra through Codex.
2. Visualize weights and results in a 3D viewer (`viz/`).
3. Run Filippos's experiments: Phi-3.5 vs MediPhi circuit discovery and transfer (EXP-05), failure-guided weight
   editing (EXP-06), training-free layer looping (EXP-07).
4. Record everything in the Obsidian vault (`vault/`).

## Established

- EXP-01 (MIT exp 3) replicated: corpus right in 6 of 6 Astra conversations, all 10 true copying heads found, one
  head over-called twice (`experiments/exp_01_mit3_language/RESULTS.md`).
- EXP-03a: MIT's published Qwen3.5-0.8B edit (p50c) rebuilt bit for bit from public weights; it is
  `W' = W - 6 u (u^T W)` on 64 columns of `down_proj` in layers 15 and 19.
- EXP-04 (letter scoring, MedQA): test 1,273 questions: Phi-3.5 0.5719, MediPhi 0.5632, MediPhi-Instruct 0.5648;
  dev 1,272: 0.5841 / 0.5951 / 0.5802; first 3,000 training questions: 0.6037 / 0.5923 / 0.5780. No medical model
  is better than the base under this scoring.
- EXP-04b (reasoning, 400 test questions): Phi-3.5 0.630, MediPhi 0.600, MediPhi-Instruct 0.578 (it answers
  directly instead of reasoning). No protocol shows a MedQA advantage for the medical models.
- EXP-04c: the paper's MedQA numbers reproduce in the bare lm-evaluation-harness format (0.4855 / 0.4808 / 0.5460
  vs 0.486 / 0.491 / 0.548); the MediPhi-Instruct gain exists only there.
- EXP-04c, practice pool (dev + first 3,000 training questions, bare prompt): Phi-3.5 0.4846, MediPhi-Instruct
  0.5421 (+5.76 points, SE 0.74; 632 questions only the donor gets, 386 only Phi-3.5 gets).
- EXP-06 `run1` (failure-guided editing, no donor, 30 passes): Astra kept no edit; test accuracy unchanged at
  0.5719. Best candidate (layer 26 MLP unit 4131): +1.95 points on its 512 selection questions, +0.68 (SE 0.54) on
  1,024 untouched ones (`experiments/exp_06_failure_guided_editing/RESULTS.md`).
- EXP-05a: MediPhi - Phi-3.5 is a small (1.57 % relative norm), dense, high-rank change spread over every
  attention and MLP matrix; MediPhi is a `breadcrumbs_ties` merge of five experts.

None of the above has been reviewed by the researcher; nothing is in `research/07_RESULTS_GROUND_TRUTH.md`.

## Decisions

- Researcher: replicate MIT experiments 3, 7, 8; this Mac first, ask before cloud; Claude runs `codex exec`;
  commit and push as work proceeds.
- Researcher (2026-11-22): donor experiment = MediPhi-Instruct into Phi-3.5 on the bare-prompt MedQA gap; success
  needs bare accuracy up and chat accuracy not down; no extra benchmarks for now.
- Claude (provisional): MedQA from `bigbio/med_qa`; letter-logit scoring; discovery pool = dev + first 3,000
  training questions, test split only for final numbers.

## Changes

- New code: `wsp/` (harness, MedQA, checkpoints, looping, patching, lab), `viz/`, `experiments/exp_01..07`, `tests/`.
- New notes: `vault/` (papers, concepts, experiments, sessions, visualizations), `research/01_PROBLEM.md`,
  `research/03_LITERATURE_LEDGER.md`, `docs/intakes/2026-11-21_1730_project-kickoff.md`.

## Verification

- `ruff check wsp experiments tests` clean; `pytest tests` 23 passed.
- `python tests/integration_lab.py <scratch>` passes on the real Phi-3.5 weights.
- Viewer builds (`npm run build --prefix viz`) and was checked in the browser for all six datasets.

## Open Issues

- MIT exp 7 needs `google/gemma-3-270m-it`, gated for the researcher's Hugging Face account.
- MIT exp 8 part C (Astra's own Qwen session) is paused: passes are slow on this Mac and MedQA has GPU priority.
- Some Codex tool definitions cannot be disabled by flag; turns are audited for tool use instead.
- Memory: keep GPU jobs behind `python -m wsp.gpu_lock` (`batch` for single-model sweeps, default `gpu` for lab
  passes; a job that loads two models takes both, `batch` first).

## Next

1. Collect: EXP-05c `transfer1` (`experiments/exp_05_medical_circuit/raw/transfer1/`, started 23:31), and the
   sweep queue `bash scripts/queue_sweeps.sh` (started 23:36): weight swaps and patching
   (`processed/patch_mediphi_instruct_bare/`), looping scan, chat-template control, Qwen behaviour check.
2. Write `RESULTS.md` for EXP-05, 07 and EXP-03b; export interventions and looping to the viewer
   (`python -m wsp.viz_export interventions looping`).
3. Restart EXP-03c with the smaller passes when the GPU is free; EXP-02 once Gemma access exists.

## Resume

> Read `vault/Home.md`, then the session note. GPU jobs are serialized with `python -m wsp.gpu_lock [--name batch] -- <cmd>`.
> Astra runs through `/Applications/ChatGPT.app/Contents/Resources/codex-cli/bin/codex` (see `vault/concepts/Astra harness.md`).
> An interrupted Astra session resumes by re-running its command (state is in `raw/<run>/state.json`).
