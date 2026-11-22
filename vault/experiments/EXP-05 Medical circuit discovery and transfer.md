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

## Parts
| Part | What | Status |
|---|---|---|
| 05a | Weight difference MediPhi - Phi-3.5 per tensor, head and neuron | done |
| 05b | Cross-model activation patching and per-layer weight swapping | queued |
| 05c | Astra session with the donor checkpoint in the edit lab | not started (waiting on the donor question) |
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

## 05b - activation patching and weight swapping (queued)
`experiments/exp_05_medical_circuit/patch.py`, tools in `wsp/patching.py` (tests in `tests/test_patching.py`).
- Question sets from the discovery pool (dev + 3,000 training questions), 60 each: `gain` (donor right, base
  wrong), `loss` (base right, donor wrong), `both` (both right).
- Activation patching: the donor's attention output, MLP output or residual stream of one layer replaces the
  base's, all positions; 96 patches per question.
- Weight swapping: one layer's attention weights or MLP weights copied from the donor into the base.
- The `loss` set is the control: if patching "recovers" as much there in the wrong direction, the effect is
  disagreement between two similar models rather than a medical gain.

## The donor question (open, needs the researcher)
Under letter scoring neither `microsoft/MediPhi` nor `microsoft/MediPhi-Instruct` is better than Phi-3.5 on
MedQA ([[EXP-04 MedQA baseline - Phi-3.5 vs MediPhi]]). There is a set of questions MediPhi gets right and
Phi gets wrong, and a set of about the same size in the other direction.

## Visualizations
Viewer dataset `weightdiff_mediphi_phi` ([[Visualization index]]).

## Links
[[Activation patching]] - [[Crosscoders]] - [[Transcoders]] - [[Tan et al. 2026 - Efficient Transcoder Adaptation]]
