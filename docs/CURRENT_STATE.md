# Current State

Checkpoint written 2026-11-23 17:10 (session 01, still in progress). Detailed log: `vault/sessions/2026-11-21 Session 01.md`.

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
- EXP-03b: MIT's edit changes behaviour here in the reported direction. 48 GSM8K problems, 16,384-token limit,
  MLX: correct 12 to 25 of 48 (14 fixed, 1 broken, p = 0.001), answers at the token limit 36 to 15
  (`experiments/exp_03_mit8_math/RESULTS.md`).
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
- EXP-05c `transfer1` (Astra with the donor, 30 passes): Astra submitted no edit; test accuracy unchanged at
  0.4870 (bare) and 0.5719 (chat). Its candidate (MediPhi-Instruct's attention weights and norms in layers 0 to 7
  plus 256 MLP units per layer, 4.25 % of the weights changed) gained +1.76 points (SE 1.32) under the bare prompt
  and -0.29 (SE 0.81) under the chat prompt on 1,024 untouched practice questions, against +3.6 points for the
  whole donor; the gain sits on questions the unedited model already answers correctly under the chat prompt
  (+6.07 points there, -5.03 elsewhere). All of Astra's figures were recounted and the saved candidate re-scored
  with identical results (`experiments/exp_05_medical_circuit/RESULTS_05c.md`).
- EXP-05c, withdrawn candidate on the test split (scored once after the session, at the researcher's request):
  bare 48.70 % to 50.98 % (+2.28 points, SE 1.10; 113 fixed, 84 broken, p = 0.046), chat 57.19 % to 56.32 %
  (-0.86 points, SE 0.75). MediPhi-Instruct itself: 54.60 % and 56.48 %. Astra's rule still fails on the chat
  condition. Confirmation and test pooled afterwards: bare +2.05 (SE 0.85), chat -0.61 (SE 0.55).
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

- `ruff check wsp experiments tests` clean; `pytest tests` 27 passed (2026-11-23 17:10).
- `python tests/integration_lab.py <scratch>` passes on the real Phi-3.5 weights.
- Viewer builds (`npm run build --prefix viz`) and was checked in the browser for all nine datasets.

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
- EXP-05c: the candidate's controls were scored neither on the confirmation questions nor on the test split, so
  its test-split gain is not shown to be specific to these weights. The saved candidate is local only
  (`raw/transfer1/work/candidate_sparse_full/`, 3.5 GB).
- Harness gap: a lab pass records the names of the edited tensors, not their content, so a withdrawn edit can be
  tied to a pass only by scoring a saved copy again (done for `transfer1`). A digest of the edits in each pass
  record would close it; not implemented.

## Next

1. Nothing is running. All queued runs and both Astra sessions on Phi-3.5 are finished and written up.
2. Where to run next: the researcher has a SLURM cluster account (login node reachable only with a password,
   so not from this session). Status pasted by the researcher on 2026-11-23: it has GPU nodes with 8 x H100 80 GB,
   4 x RTX A6000 48 GB + 6 x RTX 2080 Ti, 4 x Titan Xp and 1 x GTX 1080 (7-day time limit), but none of them is
   schedulable right now (states down, drain or invalid; one CPU-only node is idle), and the home filesystem is
   100 % full with 0 bytes available. So it cannot run these jobs until its administrators bring nodes back and
   there is somewhere to write. Until then: this Mac (lid open, on power), or another cluster or rented GPU if
   the researcher has one. Astra itself must stay on the Mac (Codex build in ChatGPT.app). `wsp/models.py` picks
   MPS or CPU only; CUDA support is a small change that has not been made.
3. Researcher decisions pending: how to read the EXP-05c test-split result (bare up, chat slightly down) and
   whether to score the two controls on the test split; whether to score the EXP-06 candidate there too; control
   and unselected-question scoring for EXP-05b; `around` sweep for EXP-07; whether the bare prompt stays the
   primary measure, given that both the chat-template control and the EXP-05c split point at prompt handling; a
   second donor or no-donor session.
4. Restart EXP-03c with the smaller passes (the GPU is free now); EXP-02 once Gemma access exists; EXP-05d
   (crosscoder) not started.

## Resume

> Read `vault/Home.md`, then the session note. GPU jobs are serialized with the locks in `wsp/gpu_lock.py` (per unit of work in the script, or `python -m wsp.gpu_lock [--name batch] -- <cmd>` for a whole command).
> Astra runs through `/Applications/ChatGPT.app/Contents/Resources/codex-cli/bin/codex` (see `vault/concepts/Astra harness.md`).
> An interrupted Astra session resumes by re-running its command (state is in `raw/<run>/state.json`).
