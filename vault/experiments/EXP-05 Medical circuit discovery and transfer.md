---
type: experiment
id: EXP-05
status: awaiting-review
tags: [experiment, weight-diff, activation-patching, medqa]
---

# EXP-05 - Medical circuit discovery and transfer (Phi-3.5 vs MediPhi)

## Question
Filippos's main direction: on MedQA questions the medical fine-tune gets right and the base gets wrong, which
layers, heads, MLPs or features are causally responsible, and can that be moved into the base model by targeted
weight edits?

## Setting (decided by the researcher, 2026-11-22)
Donor = `microsoft/MediPhi-Instruct`. Primary measure = MedQA with the bare lm-evaluation-harness prompt, where
the donor beats Phi-3.5 (test: 54.6 % vs 48.5 %). Guardrail = the chat prompt, where they tie; an edit must not
lower it. Background: [[EXP-04 MedQA baseline - Phi-3.5 vs MediPhi]].

## Parts
| Part | What | Status |
|---|---|---|
| 05a | Weight difference MediPhi - Phi-3.5 per tensor, head and neuron | done (for `microsoft/MediPhi`) |
| 05b | Cross-model activation patching and per-layer weight swapping, MediPhi-Instruct, bare prompt | done 2026-11-23, awaiting review |
| 05c | Astra session with the donor in the edit lab (`transfer1`, 30 passes) | done 2026-11-23 14:22, no edit submitted; withdrawn candidate scored on the test split 16:58; awaiting review |
| 05d | Crosscoder on one residual layer of both models | not started |

## 05a - weight difference (done)
`experiments/exp_05_medical_circuit/weight_diff.py` -> `processed/weight_diff_summary.json`, `weight_diff_maps.npz`.
- Whole model: the change has 1.57 % of the norm of the base weights.
- Every attention and MLP matrix in all 32 layers differs; 50-65 % of entries changed.
- Broad spectrum: 90 % of the change's energy needs about 1,500-2,400 of 3,072 singular values, so the change is
  not a low-rank adapter. Stable rank 150-530.
- Normalization gains are identical to the base except `input_layernorm` in layers 0-2 and
  `post_attention_layernorm` in layer 0.
- Relative change by depth: attention output (o_proj) 4.9 % at layer 0 falling to 0.8 % at layer 31;
  query/key/value 0.1 % rising to 1.5 %; MLP 0.7 % at layer 0, then 1.4-1.8 % everywhere.
- Largest per-head change: output of head 1.4 (11.4 %); no head or neuron stands far apart from the rest.
- The model repo ships `mergekit_config.yml`: `breadcrumbs_ties` over five experts (PubMed, Clinical, MedCode,
  MedWiki, Guidelines), densities 0.14-0.93, with Phi-3.5 as the base. See [[Corbeil et al. 2025 - MediPhi]].

## 05b - activation patching and weight swapping (done 2026-11-23)
`experiments/exp_05_medical_circuit/patch.py --donor mediphi_instruct --style bare`, tools in `wsp/patching.py`.
- Question sets from the discovery pool (dev + 3,000 training questions), 60 each, defined with the bare prompt:
  `gain` (donor right, base wrong), `loss` (base right, donor wrong), `both` (both right).
- Activation patching: the donor's attention output, MLP output or residual stream of one layer replaces the
  base's, all positions; 96 patches per question.
- Weight swapping: one layer's attention weights or MLP weights copied from the donor into the base.
- The `loss` set is the control: if patching moves margins as much there in the wrong direction, the effect is
  disagreement between two similar models rather than a transferable gain.
- Output: `processed/patch_mediphi_instruct_bare/`. Handoff: `experiments/exp_05_medical_circuit/RESULTS.md`.

### 05b results (factual, awaiting review)
Means over 60 questions per set, margin in log-probability units.
- Before any change: gain set Phi-3.5 -1.74 / MediPhi-Instruct +1.17 (difference +2.91); loss set +1.51 / -0.81
  (-2.31); both set +3.22 / +2.30 (-0.93).
- `[OBSERVATION]` Residual stream from MediPhi-Instruct after layer L (= its layers 0 to L, then Phi-3.5's layers
  L+1 to 31): the gain set's difference is closed to 7 %, 28 %, 50 %, 72 %, 86 %, 97 % after layers 0, 2, 4, 6, 8,
  10, and stays between 97 % and 105 % for every later layer. The loss set falls by the same share. So Phi-3.5's
  layers 11 to 31 give MediPhi-Instruct's answers when they are fed MediPhi-Instruct's stream.
- `[OBSERVATION]` One layer's weights: the largest effect on the gain set is +0.244 (attention weights of layer 1),
  8 % of the +2.92 for all weights; it turns 10 of 60 gain questions right and 9 of 60 loss questions wrong. The 32
  attention swaps add up to +1.50 (61 % of it from layers 0 to 8), the MLP swaps to +0.39.
- `[OBSERVATION]` One attention or MLP output: at most 50 % of the difference (attention output of layer 25,
  +1.45, 25 of 60 right), with effects of both signs in late layers (attention 31: -0.89, MLP 28: -0.99).
- `[OBSERVATION]` What raises the gain set mostly lowers the loss set (correlation -0.51 over the 64 weight swaps).
  A small exception: MLP weights of layer 6, +0.106 (SE 0.018) on gain, -0.007 on loss.
- Not established: that any of this improves accuracy (the sets are selected by outcome and nothing was scored on
  unselected questions); that the effect is specific to the donor (no control swap of equal size); how much is
  medical content and how much is handling a prompt without a chat template
  ([[EXP-04 MedQA baseline - Phi-3.5 vs MediPhi]]: with the template around the same text the two models tie).

## 05c - Astra with the donor (done 2026-11-23)
`experiments/exp_05_medical_circuit/run_astra.py --name transfer1 --donor mediphi_instruct --budget 30`.
- Lab: `wsp/lab.py` with `style="bare"`; task message `transfer_message` in `wsp/lab_session.py`.
- Astra has both sets of weights (`load_donor`), practice results of both models under both prompts, and three
  kinds of forward pass: score (bare or chat), teacher-forced reads, and activation patching (donor into the base,
  per layer component or per head, up to 64 questions x 256 patches).
- The message asks Astra to say whether the advantage is about the prompt format or about medical content, and to
  report the share of weights its final edit changes.
- Final measurement: original vs edited Phi-3.5 on the MedQA test split under both prompts
  (`raw/transfer1/final_eval.json`).
- Handoff: `experiments/exp_05_medical_circuit/RESULTS_05c.md`. Checks: `check_session.py` ->
  `processed/check_transfer1.json`, `check_transfer1_questions.jsonl`.

### 05c results (factual, awaiting review)
Session: 2026-11-22 23:31 to 2026-11-23 14:09, 69 replies, 30 forward passes (22 score, 5 teacher-forced, 3
patching). Astra split the 4,272 practice questions itself: 1,024 set aside for confirmation, the rest for
development (64 for patching, 256 to compare transfers, 512 as a second check).
- `[OBSERVATION]` Astra submitted no edit. Test split, 1,273 questions: 48.70 % (bare) and 57.19 % (chat) before
  and after.
- The candidate it built and then withdrew: MediPhi-Instruct's attention weights and norm vectors in layers 0 to
  7, plus 256 of 8,192 MLP units per layer. 8.40 % of the model's weights are in the transfer, 4.25 % differ.
- `[OBSERVATION]` Candidate against the unedited model, bare prompt: +4.30 points (SE 2.23) on the 256 questions
  used to choose it, +1.95 (SE 1.68) on 512 fresh development questions, +1.76 (SE 1.32; 100 fixed, 82 broken) on
  the 1,024 confirmation questions. Chat prompt: -0.39, 0.00, -0.29 points. MediPhi-Instruct itself is +3.6 points
  above Phi-3.5 on those confirmation questions.
- Astra's rule, written before the confirmation passes: keep the edit only if the bare-prompt gain exceeds 2
  standard errors and chat accuracy does not fall. Neither held.
- `[OBSERVATION]` Test split (1,273 questions), scored once after the session at the researcher's request:
  bare prompt 48.70 % to 50.98 %, +2.28 points (SE 1.10; 113 fixed, 84 broken, sign test p = 0.046); chat prompt
  57.19 % to 56.32 %, -0.86 points (SE 0.75; 40 fixed, 51 broken). MediPhi-Instruct itself: 54.60 % and 56.48 %.
  By Astra's rule the bare gain now passes (2.07 standard errors) and the chat condition still fails.
- Confirmation and test together (2,297 questions; pooled afterwards, not planned): bare +2.05 points (SE 0.85),
  chat -0.61 points (SE 0.55).
- `[OBSERVATION]` On the confirmation questions the bare-prompt gain sits where the unedited model is already
  right under the chat prompt: +6.07 points (SE 1.73) on those 626 questions, -5.03 (SE 1.96) on the other 398.
  On the test split: +4.67 (SE 1.50) on 728 questions, -0.92 (SE 1.61) on the other 545.
- `[OBSERVATION]` The candidate raises the probability the model puts on the four answer letters under the bare
  prompt from 0.043 to 0.134 (MediPhi-Instruct: 0.413).
- `[OBSERVATION]` Copying the weights of the components with the largest patching effects did nothing: the whole
  MLP of layer 11 +0.78 points, six attention heads +0.39 points (256 questions). Every transfer above +2 points
  contains the attention weights of layers 0 to 7.
- `[OBSERVATION]` Copying every weight of layers 0 to 7 gave the same accuracy as the candidate. The candidate at
  1.5 times strength gave +4.30 points (SE 1.97) under the bare prompt and -0.98 under the chat prompt.
- `[OBSERVATION]` Astra's final answer reads the result as "prompt adaptation and extraction of existing answers,
  not demonstrated acquisition of medical knowledge". That is Astra's reading, not a reviewed conclusion.
- Checked by Claude: all 22 score passes recount exactly from the lab's files; the confirmation questions appear
  in no earlier pass; the saved candidate contains only donor values inside the transfer and base values outside;
  scoring the saved candidate again reproduces both confirmation passes with no difference on any question.
- Not established: whether the bare-prompt gain is real (two untouched estimates, +1.76 and +2.28, each about
  1.3 to 2.1 standard errors); whether the chat prompt loses about a point or nothing; that the candidate beats a
  control of the same size (no control was scored on untouched questions); medical content against prompt
  handling.

## Visualizations
Viewer datasets `weightdiff_mediphi_phi`, `interventions_mediphi_instruct_bare_phi` and
`exp05c_transfer_transfer1` ([[Visualization index]]).

![[viz_exp05b_patching_activations.jpg]]

![[viz_exp05b_patching_weights.jpg]]

![[viz_exp05c_transfers.jpg]]

![[viz_exp05c_candidate_weights.jpg]]

![[viz_exp05c_per_question.jpg]]

## Links
[[Activation patching]] - [[Crosscoders]] - [[Transcoders]] - [[Tan et al. 2026 - Efficient Transcoder Adaptation]]
