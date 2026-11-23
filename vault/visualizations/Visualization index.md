---
type: visualization-index
tags: [visualization]
---

# Visualization index

All interactive views live in the viewer under `viz/` (see `viz/README.md`): run
`python -m wsp.viz_export mit3 mit8 weightdiff exp06`, then `npm run dev --prefix viz`, and open http://localhost:5173.

## MIT experiment 3: the blocks Astra read
Datasets `mit3_python`, `mit3_romaji`, `mit3_english`. Each of the four blocks shows LayerNorm gains, W_q,
W_k, W_v, W_o, W_fc and W_proj cell by cell. Head boundaries are drawn as separators. Marked heads:
green = named a copying head by Astra and confirmed by MIT's measurement, yellow = named by Astra but not
confirmed, orange = measured but missed.

![[viz_mit3_python_block3.jpg]]
Block 3 of the Python network: heads 1 and 5 are confirmed copying heads, head 0 is the one Astra over-called.

![[viz_mit3_python_block3_Wq_zoom.jpg]]
Zoomed into W_q with the hover readout (row, head, column, value, scale).

Source: [[EXP-01 MIT exp 3 - reading a language model's middle blocks]].

## Phi-3.5 vs MediPhi: where the weights differ
Dataset `weightdiff_mediphi_phi`. For each of the 32 layers: a 4 x 32 strip of relative change per attention
head (rows Q, K, V, O) and three 64 x 128 tiles of relative change per MLP neuron (gate, up, down).

![[viz_weightdiff_tower.jpg]]

Source: [[EXP-05 Medical circuit discovery and transfer]].

## MIT experiment 8: the Qwen3.5-0.8B edit
Dataset `mit8_qwen_p50c`. For layers 15 and 19: which of the 3,584 MLP output columns were edited, the matrix
before the edit, the change, and a close-up of the 64 edited columns before / after / change next to the loop
direction u that was removed from them.

![[viz_mit8_qwen_edit.jpg]]

Source: [[EXP-03 MIT exp 8 - editing Qwen3.5-0.8B for math]].

## EXP-06: the one unit Astra tried to change in Phi-3.5
Dataset `exp06_edit_run1`. Layer 26's MLP output: which unit was edited, each unit's direct push on answer A, the
unit's output direction before / kept part / reversed part / after. Then one cell per practice question for the
512 questions used to choose the edit, 256 reserve questions and 1,024 confirmation questions: margin before,
outcome (red = fixed, blue = broken), change of the margin.

![[viz_exp06_run1_unit.jpg]]

![[viz_exp06_run1_per_question.jpg]]
Selection: 12 fixed, 2 broken. Confirmation: 19 fixed, 12 broken. The right column shows the edit pushing wrong
answers further down (blue, top) and right answers further up (red, bottom).

Source: [[EXP-06 Failure-guided weight editing]].

## Planned
- Activation patching and weight-swap maps (layer x component), MediPhi-Instruct into Phi-3.5, bare prompt
  (`python -m wsp.viz_export interventions` once `processed/patch_mediphi_instruct_bare/` exists).
- Astra's donor edit on Phi-3.5 (EXP-05c), before and after.
- Layer-looping sweep (`python -m wsp.viz_export looping`).
