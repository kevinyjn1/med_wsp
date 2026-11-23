# Current State

Checkpoint written 2026-11-23 12:00 (session 01, still in progress). Detailed log: `vault/sessions/2026-11-21 Session 01.md`.

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
- EXP-04c control (2026-11-23): the bare harness text inside the chat template gives 0.5530 / 0.5522 / 0.5562
  (Phi-3.5 / MediPhi / MediPhi-Instruct); the template alone removes MediPhi-Instruct's lead.
- EXP-05b (activation patching and weight swaps, MediPhi-Instruct into Phi-3.5, bare prompt, 60 questions per
  set): the donor's residual stream after layer 10 or later makes Phi-3.5's remaining layers reproduce the donor's
  margins; no single layer's weights carry the difference (largest swap +0.24 of +2.92, attention weights of
  layer 1); raising the gain set mostly lowers the loss set (`experiments/exp_05_medical_circuit/RESULTS.md`).
- EXP-07 first scan (Phi-3.5, 600 dev questions, 28 looping configurations): none raises accuracy with p < 0.05;
  best is the damped rule on layers 10-13, +1.83 points (p = 0.19), margin +0.63 (SE 0.19); plain repetition of
  early layers costs 3.8 to 7.0 points (`experiments/exp_07_layer_looping/RESULTS.md`).
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

- `ruff check wsp experiments tests` clean; `pytest tests` 27 passed.
- `python tests/integration_lab.py <scratch>` passes on the real Phi-3.5 weights.
- Viewer builds (`npm run build --prefix viz`) and was checked in the browser for all eight datasets.

## Open Issues

- MIT exp 7 needs `google/gemma-3-270m-it`, gated for the researcher's Hugging Face account.
- MIT exp 8 part C (Astra's own Qwen session) is paused: passes are slow on this Mac and MedQA has GPU priority.
- Some Codex tool definitions cannot be disabled by flag; turns are audited for tool use instead.
- Memory: keep GPU jobs behind the locks in `wsp/gpu_lock.py` (`batch` for single-model sweeps, default `gpu` for
  lab passes; a job that loads two models takes both, `batch` first). Take a lock per unit of work (one
  configuration, one question), not for a whole sweep, so a lab pass can run in between.
- Power: the Mac is on a 65 W adapter and these jobs draw more, so the battery drains while plugged in and macOS
  sleeps when it is empty. Jobs now hold a no-sleep assertion on AC power (`wsp/keep_awake.py`), which does not
  help once the battery is empty. A closed lid always sleeps.
- EXP-05b has no control swap and no score on unselected questions; EXP-07's best window is unconfirmed.

## Next

1. Collect when they end: EXP-05c `transfer1` (`experiments/exp_05_medical_circuit/raw/transfer1/`, 14 of 30
   passes at 12:00 on 11-23; final scores in `final_eval.json`) and the Qwen behaviour check (EXP-03b,
   `experiments/exp_03_mit8_math/processed/eval_<model>_gsm8k_n48_t16384.jsonl`, last two steps of the queue).
2. Then: add EXP-05c to `experiments/exp_05_medical_circuit/RESULTS.md`, write the EXP-03b handoff, export the
   donor edit to the viewer, redact and commit `raw/transfer1`, and change `scripts/queue_sweeps.sh` to call the
   looping and patching scripts without the lock wrapper (they take the lock per unit themselves now).
3. Researcher decisions pending: control and unselected-question scoring for EXP-05b; `around` sweep for EXP-07;
   whether the bare prompt stays the primary measure; a second no-donor session (EXP-06).
4. Restart EXP-03c with the smaller passes when the GPU is free; EXP-02 once Gemma access exists.

## Resume

> Read `vault/Home.md`, then the session note. GPU jobs are serialized with the locks in `wsp/gpu_lock.py` (per unit of work in the script, or `python -m wsp.gpu_lock [--name batch] -- <cmd>` for a whole command).
> Astra runs through `/Applications/ChatGPT.app/Contents/Resources/codex-cli/bin/codex` (see `vault/concepts/Astra harness.md`).
> An interrupted Astra session resumes by re-running its command (state is in `raw/<run>/state.json`).
