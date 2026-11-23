# EXP-05b Results - Which layers carry the MedQA difference between Phi-3.5 and MediPhi-Instruct (bare prompt)

This file covers part b of EXP-05 (activation patching and per-layer weight swapping). Part a (weight
difference) is recorded in `vault/experiments/EXP-05 Medical circuit discovery and transfer.md` and has no
handoff file yet. Part c (the Astra donor session `transfer1`) is still running and will be added when it ends.

## Experiment
Filippos Bellos's main direction (2026-11-21): on MedQA questions the medical fine-tune answers correctly and the
base model does not, find which layers are causally responsible. No formal experiment contract was written. The
working specification is the vault note above and the docstring of `patch.py`. The setting was decided by the
researcher on 2026-11-22: donor = `microsoft/MediPhi-Instruct`, prompt = the bare lm-evaluation-harness prompt
(the only format in which the donor beats Phi-3.5, see EXP-04 Protocol C). Set sizes, the patch list and the seed
are Claude's provisional choices.

## Run status
COMPLETE. All 180 questions, all 96 activation patches per question, all 66 weight configurations.

## What actually ran
- Base model (receives every patch): `microsoft/Phi-3.5-mini-instruct`. Donor: `microsoft/MediPhi-Instruct`.
  Both float16 on Apple MPS, 32 layers.
- Prompt: "Question: ...\nA. ...\nB. ...\nC. ...\nD. ...\nAnswer:" as plain text, no chat template
  (`style="bare"` in `wsp/medqa.py`). Score = next-token log-probabilities of the four answer letters.
  Margin = log-probability of the correct letter minus the best wrong letter; above zero means answered correctly.
- Question pool: MedQA dev (1,272) + the first 3,000 training questions, scored with the bare prompt for both
  models (`exp_04_medqa_baseline/processed/bare_*`). Outcomes in the pool of 4,272: both right 1,684, only
  MediPhi-Instruct right 632, only Phi-3.5 right 386, neither 1,570.
- Question sets, 60 each, sampled with seed 0 (`processed/patch_mediphi_instruct_bare/question_sets.json`):

  | Set | Definition | From dev / training |
  |---|---|---|
  | gain | only MediPhi-Instruct right | 16 / 44 |
  | loss | only Phi-3.5 right | 17 / 43 |
  | both | both right | 28 / 32 |

- Activation patching (`--mode activation`): per question, one donor pass to record activations, one unpatched
  base pass, then 96 patched base passes, one per layer and component. Components: the attention block's output,
  the MLP block's output, and the residual stream after the layer. A patch replaces the base model's activation at
  every token position. A residual-stream patch after layer L is therefore the same as running the donor's layers
  0 to L and then the base model's layers L+1 to 31. One question per pass, 24 patches per batch. 4,127 s,
  2026-11-23 01:11 to 02:20.
- Weight swapping (`--mode weights`): one layer's attention weights (`qkv_proj`, `o_proj`, `input_layernorm`) or
  MLP weights (`gate_up_proj`, `down_proj`, `post_attention_layernorm`) copied from the donor into the base model,
  everything else left as Phi-3.5; plus `base` (no swap) and `all` (every weight copied). Scored in batches of 16.
  2026-11-22 23:32 to 2026-11-23 00:41.

## Deviations from contract
- No contract exists.
- None from the working specification.

## Artifacts
- `processed/patch_mediphi_instruct_bare/question_sets.json` - the 180 question ids by set.
- `processed/patch_mediphi_instruct_bare/activation.jsonl` - one line per question: base margin, donor margin, 96
  patched margins.
- `processed/patch_mediphi_instruct_bare/weights.jsonl` - one line per weight configuration: the margin of every
  question and the share answered correctly per set.
- Logs (not committed): `logs/exp05_patch_bare_weights.log`, `logs/exp05_patch_bare_activation.log`.
- Viewer dataset `interventions_mediphi_instruct_bare_phi` (`python -m wsp.viz_export interventions`), screenshots
  `vault/attachments/viz_exp05b_patching_activations.jpg` and `viz_exp05b_patching_weights.jpg`.

## Numerical results
All margins are means over the 60 questions of a set, in log-probability units, with the standard error of the
mean in brackets. "Right" is the number of the 60 questions answered correctly.

### The three sets before any intervention (activation run)
| Set | Phi-3.5 margin | MediPhi-Instruct margin | Difference | Phi-3.5 right | MediPhi-Instruct right |
|---|---|---|---|---|---|
| gain | -1.744 | +1.169 | +2.912 (0.203) | 0 | 60 |
| loss | +1.505 | -0.806 | -2.311 (0.167) | 59 | 0 |
| both | +3.222 | +2.296 | -0.926 (0.216) | 60 | 60 |

### Residual stream taken from MediPhi-Instruct after layer L
Change of Phi-3.5's margin, and questions right afterwards.

| After layer | gain: change | gain: right | loss: change | loss: right | both: change | both: right |
|---|---|---|---|---|---|---|
| 0 | +0.21 | 9 | -0.13 | 54 | +0.03 | 60 |
| 2 | +0.81 | 17 | -0.70 | 44 | -0.11 | 59 |
| 4 | +1.460 (0.165) | 28 | -1.219 (0.174) | 35 | -0.153 (0.152) | 59 |
| 6 | +2.09 | 38 | -1.60 | 25 | -0.20 | 58 |
| 8 | +2.51 | 44 | -2.08 | 15 | -0.37 | 56 |
| 10 | +2.827 (0.198) | 50 | -2.220 (0.182) | 14 | -0.371 (0.215) | 57 |
| 12 | +3.020 (0.210) | 54 | -2.398 (0.184) | 11 | -0.536 (0.233) | 58 |
| 16 | +3.02 | 57 | -2.43 | 2 | -0.71 | 59 |
| 20 | +2.88 | 57 | -2.32 | 2 | -0.95 | 60 |
| 31 | +2.894 (0.205) | 59 | -2.354 (0.169) | 0 | -0.933 (0.217) | 60 |

As a share of the gain set's difference of +2.912: 50 % after layer 4, 97 % after layer 10, and between 97 % and
105 % for every layer from 10 to 31 (maximum +3.07 after layers 14 and 15).

### One attention output or one MLP output taken from MediPhi-Instruct
The five largest mean changes per set and component (layer: change).

| Set | Attention output | MLP output |
|---|---|---|
| gain | 25: +1.45, 18: +1.28, 19: +0.96, 21: +0.93, 31: -0.89 | 12: +1.00, 28: -0.99, 26: +0.82, 4: +0.79, 10: +0.75 |
| loss | 28: -1.00, 21: -0.77, 18: -0.76, 25: -0.66, 6: -0.65 | 17: -1.02, 4: -0.89, 19: -0.81, 15: -0.79, 20: -0.77 |
| both | 31: -0.83, 11: -0.62, 22: -0.60, 6: -0.51, 18: -0.48 | 31: -1.28, 19: -0.94, 14: -0.91, 20: -0.87, 15: -0.86 |

Selected entries with standard errors and counts: gain, attention 25: +1.446 (0.172), 25 right; gain, attention
18: +1.283 (0.125), 22 right; gain, MLP 12: +1.001 (0.154), 19 right; gain, MLP 28: -0.988 (0.229), 5 right;
loss, attention 28: -0.997 (0.146), 41 right; loss, MLP 17: -1.024 (0.169), 38 right. The most questions any
single attention-output or MLP-output patch turns right in the gain set is 25 of 60 (attention, layer 25).

### One layer's weights copied from MediPhi-Instruct
| Configuration | gain: change | gain: right | loss: change | loss: right | both: change | both: right |
|---|---|---|---|---|---|---|
| none (Phi-3.5) | 0 | 0 | 0 | 60 | 0 | 60 |
| attention weights, layer 0 | +0.217 (0.042) | 10 | -0.101 (0.042) | 55 | +0.063 (0.039) | 60 |
| attention weights, layer 1 | +0.244 (0.058) | 10 | -0.180 (0.058) | 51 | +0.122 (0.036) | 60 |
| attention weights, layer 2 | +0.142 (0.064) | 8 | -0.171 (0.053) | 56 | -0.108 (0.047) | 56 |
| attention weights, layer 4 | +0.080 (0.050) | 9 | -0.175 (0.044) | 56 | -0.132 (0.039) | 58 |
| attention weights, layer 8 | -0.021 (0.038) | 5 | -0.139 (0.040) | 55 | -0.203 (0.031) | 59 |
| MLP weights, layer 2 | +0.048 (0.048) | 6 | -0.179 (0.051) | 53 | -0.146 (0.035) | 59 |
| MLP weights, layer 6 | +0.106 (0.018) | 4 | -0.007 (0.021) | 60 | +0.031 (0.016) | 59 |
| MLP weights, layer 7 | +0.111 (0.026) | 5 | -0.077 (0.023) | 59 | -0.009 (0.021) | 60 |
| every weight | +2.915 (0.204) | 60 | -2.313 (0.166) | 1 | -0.927 (0.217) | 60 |

- Largest single-layer effect on the gain set: +0.244 (attention weights of layer 1), 8 % of the +2.915 that
  copying every weight gives. From layer 16 on, no attention swap exceeds 0.07 in absolute value on the gain set,
  and no MLP swap exceeds 0.12 at any layer.
- Sum of the 32 single-layer effects on the gain set: attention +1.503 (layers 0 to 8: +0.921, layers 9 to 31:
  +0.582), MLP +0.389 (layers 0 to 8: +0.297, layers 9 to 31: +0.092). Together +1.892, against +2.915 for all
  weights at once.
- Loss set, sum of single-layer effects: attention -1.664, MLP -0.736. Both set: attention -1.155, MLP -0.586.

## Observations
- `[OBSERVATION]` With the donor's residual stream after layer 10 or any later layer, Phi-3.5's remaining layers
  reproduce the donor's margin on the gain set (97 % to 105 % of the difference) and on the loss set (96 % to
  107 %, 0 to 14 of 60 still right). On the both set the difference closes later: 40 % after layer 10, 77 % after
  layer 16, and 94 % to 106 % from layer 19 on.
- `[OBSERVATION]` The share of the gain set's difference closed by the donor's lower layers rises steadily over
  layers 0 to 10 (7 %, 28 %, 50 %, 72 %, 86 %, 97 % after layers 0, 2, 4, 6, 8, 10). The largest step from one
  layer to the next is 0.417 margin units (layer 4).
- `[OBSERVATION]` No single layer's weights carry the difference. The largest single-layer weight swap moves the
  gain set by 8 % of the full difference and turns 10 of 60 questions right, while the same swap turns 9 of 60
  loss questions wrong.
- `[OBSERVATION]` The single-layer weight effects are concentrated in the attention weights of the first nine
  layers (61 % of the summed attention effect on the gain set comes from layers 0 to 8).
- `[OBSERVATION]` The interventions with the largest effect on the gain set also lower the loss set. Residual
  stream after layer 12: +3.02 and -2.40. Attention output of layer 25: +1.45 and -0.66. Attention weights of
  layer 1: +0.24 and -0.18. Over the 64 single-layer weight swaps the correlation between the gain set's change
  and the loss set's change is -0.51; over the 64 attention-output and MLP-output patches it is -0.44.
- `[OBSERVATION]` Exceptions to that pattern are small. MLP weights of layer 6 raise the gain set by +0.106
  (0.018) and change the loss set by -0.007 (0.021) and the both set by +0.031 (0.016).
- `[OBSERVATION]` Single attention-output and MLP-output patches have effects of both signs in late layers on
  the gain set (attention 25: +1.45, attention 31: -0.89, MLP 28: -0.99).

## Negative / null results
- No single-layer weight swap turns more than 10 of the 60 gain questions right.
- No single attention-output or MLP-output patch closes more than 50 % of the gain set's difference.

## Failed / excluded runs
- None for these two runs. An earlier launch of the whole sweep queue at 2026-11-22 23:31 failed before any model
  was loaded (shell quoting error, exit 127) and produced no output.

## Anomalies
- One question is scored differently by the two runs. `dev-01195` (loss set) has a Phi-3.5 margin of 0.0 in the
  activation run (one question per pass) and +0.0625 in the weight run (batches of 16), so Phi-3.5 has 59 of 60
  loss questions right in one run and 60 in the other. Across all 180 questions the two runs' base margins differ
  by 0.020 on average and at most 0.094 (float16 arithmetic with different batch shapes).
- With every weight copied, 1 of 60 loss questions is still right (`train-02109`, margin +0.031), although the
  donor answers all 60 wrong by definition of the set. The donor's own margins (one question per pass) and the
  all-weights margins (batches of 16) differ by 0.015 on average and at most 0.0625.

## Scope
- Models: Phi-3.5-mini-instruct and MediPhi-Instruct only. Prompt: bare harness prompt only. Task: MedQA 4-option
  questions from the practice pool; the test split was not used.
- 60 questions per set, one sample (seed 0). Whole-layer components only: no single heads, no single MLP units,
  no token-position selection.
- Each intervention changes one layer. No combinations of layers were tested except "every weight".

## What this experiment does NOT establish
- It does not show that any edit improves Phi-3.5's accuracy. The sets were selected by the two models' outcomes,
  so the gain set can only go up and the loss set can only go down. No intervention was scored on an unselected
  sample of questions.
- It does not separate a donor-specific effect from the effect of any perturbation of the same size: there is no
  control swap (for example a random change of equal norm, or weights from `microsoft/MediPhi`). Questions
  selected for being answered wrong tend to move up under any change to the model.
- It does not show which part of the difference is about medical content and which is about handling a prompt
  without a chat template. EXP-04 Protocol C shows that the accuracy difference between the two models disappears
  when the same text is wrapped in the chat template.
- It does not identify heads, units or features, and it does not test whether the layer-wise pattern holds for
  the chat prompt or for the test split.
- Single-layer effects are not additive (their sum is +1.89 against +2.92), so the table does not give the effect
  of copying a group of layers.

## Reproducibility
- Commit at run time: `98112a8` (`patch.py` and `wsp/patching.py` last changed in `addec46`).
- Commands (run by `scripts/queue_sweeps.sh`):
  `python -m wsp.gpu_lock --name batch -- python experiments/exp_05_medical_circuit/patch.py --donor mediphi_instruct --style bare --per-set 60 --mode weights`
  and the same with `--mode activation` under both GPU locks.
- Seed: 0 for the question sample. No other randomness.
- Environment: Apple M-series, MPS, float16, torch and transformers versions as in `uv.lock`.
- Both modes resume from their output files, so a re-run with the files present does nothing.

## Status
Awaiting researcher review. Not promoted to `research/07_RESULTS_GROUND_TRUTH.md`.
