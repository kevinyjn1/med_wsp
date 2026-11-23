---
type: visualization-index
tags: [visualization]
---

# Visualization index

All interactive views live in the viewer under `viz/` (see `viz/README.md`): run
`python -m wsp.viz_export mit3 mit8 weightdiff exp06 interventions looping exp05c`, then `npm run dev --prefix viz`, and open http://localhost:5173.

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

## EXP-05b: which layers carry the difference between Phi-3.5 and MediPhi-Instruct
Dataset `interventions_mediphi_instruct_bare_phi`. Columns are the 32 layers. Rows are the three question sets
(gain = only MediPhi-Instruct right, loss = only Phi-3.5 right, both = both right) times the component that was
taken from MediPhi-Instruct. Each group has two panels: the mean change of the correct answer's margin (red = up,
blue = down) and the share of the set's 60 questions answered correctly afterwards.

![[viz_exp05b_patching_activations.jpg]]
Activations. Top rows of each set: the residual stream. With MediPhi-Instruct's stream after layer 10 or later,
Phi-3.5's remaining layers give MediPhi-Instruct's answers (gain set up, loss set down by the full difference).

![[viz_exp05b_patching_weights.jpg]]
Weights. Copying one layer's weights moves the margin by at most 0.24 (attention weights of layer 1) against 2.92
for all weights; the effect sits in the attention weights of the first layers.

Source: [[EXP-05 Medical circuit discovery and transfer]].

## EXP-07: which 4-layer window to repeat
Dataset `looping_phi_dev_600`. Columns are the repeated layers (2-5 to 26-29), rows are the damped rule and plain
repetition. Three panels: change of accuracy, change of the correct answer's margin, number of questions whose
result changed.

![[viz_exp07_looping_scan.jpg]]
Plain repetition of early layers costs up to 7 points. The damped rule on layers 10-13 is the only cell with a
clear margin gain (+0.63, SE 0.19); its accuracy gain of +1.83 points has p = 0.19.

Source: [[EXP-07 Training-free layer looping]].

## EXP-05c: the weight transfer Astra built and withdrew
Dataset `exp05c_transfer_transfer1`. Three levels.

![[viz_exp05c_transfers.jpg]]
Every transfer Astra scored, one per row in the order it tried them. Left: in which layers it took
MediPhi-Instruct's weights (brighter = more of the layer's six weight tensors). Middle and right: the change of
accuracy and of the correct answer's margin on Astra's three question sets and on the test split, under both
prompts (hatched = not measured). The candidate is the row with all eight cells filled; the last two columns
(1,273) are the test split, scored after the session.

![[viz_exp05c_candidate_weights.jpg]]
The candidate's weights. Top: how much MediPhi-Instruct differs from Phi-3.5 in each layer and tensor. Below: how
much of that the candidate takes (attention and norms of layers 0 to 7 in full, the MLP only through 256 units per
layer, shown as the eight dotted tiles).

![[viz_exp05c_per_question.jpg]]
One cell per question, sorted by the unedited model's margin. On the 1,024 confirmation questions the candidate
fixes 100 and breaks 82 under the bare prompt, and fixes 33 and breaks 36 under the chat prompt; on the 1,273
test questions 113 and 84, and 40 and 51. The last column of those rows shows MediPhi-Instruct itself on the same
questions. Bottom two rows: the bare-prompt outcome with the questions ordered by the unedited model's chat
margin (fixed answers sit mostly where the chat answer is right).

Source: [[EXP-05 Medical circuit discovery and transfer]].
