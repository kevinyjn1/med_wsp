---
type: experiment
id: EXP-05
status: running
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
| 05c | Astra session with the donor in the edit lab (`transfer1`, 30 passes) | running since 2026-11-22 23:31 (10 passes used at 11:20 on 11-23) |
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

## 05c - Astra with the donor (queued)
`experiments/exp_05_medical_circuit/run_astra.py --name transfer1 --donor mediphi_instruct --budget 30`.
- Lab: `wsp/lab.py` with `style="bare"`; task message `transfer_message` in `wsp/lab_session.py`.
- Astra has both sets of weights (`load_donor`), practice results of both models under both prompts, and three
  kinds of forward pass: score (bare or chat), teacher-forced reads, and activation patching (donor into the base,
  per layer component or per head, up to 64 questions x 256 patches).
- The message asks Astra to say whether the advantage is about the prompt format or about medical content, and to
  report the share of weights its final edit changes.
- Final measurement: original vs edited Phi-3.5 on the MedQA test split under both prompts
  (`raw/transfer1/final_eval.json`).

## Visualizations
Viewer datasets `weightdiff_mediphi_phi` and `interventions_mediphi_instruct_bare_phi` ([[Visualization index]]).

![[viz_exp05b_patching_activations.jpg]]

![[viz_exp05b_patching_weights.jpg]]

## Links
[[Activation patching]] - [[Crosscoders]] - [[Transcoders]] - [[Tan et al. 2026 - Efficient Transcoder Adaptation]]
