---
type: concept
title: "Crosscoders"
tags: [concept, crosscoders, model-diffing, dictionary-learning, fine-tuning]
---

# Crosscoders

Scope: written only from [[Lindsey et al. 2024 - Sparse Crosscoders]], a self-described preliminary research update. "(inference)" marks my own reading.

## Definition
A sparse crosscoder is one dictionary whose features are computed from, and reconstruct, several activation spaces at once: several layers of one model, or matching layers of several models. "Where autoencoders encode and predict activations at a single layer, and transcoders use activations from one layer to predict the next, a crosscoder reads and writes to multiple layers." SAEs and [[Transcoders]] are special cases. A feature is "the same" across spaces if it fires on the same datapoints; its decoder direction may differ per space.

## Loss
For spaces $l \in L$ and one token position $x_j$:

$f(x_j) = \mathrm{ReLU}\left(\sum_{l\in L} W^l_{enc}\, a^l(x_j) + b_{enc}\right)$

$a^{l'}(x_j) = W^l_{dec}\, f(x_j) + b^l_{dec}$

$L = \sum_{l\in L} \lVert a^l(x_j) - a^{l'}(x_j) \rVert^2 + \sum_i f_i(x_j) \left(\sum_{l\in L} \lVert W^l_{dec,i} \rVert\right)$

The sparsity penalty on feature $i$ is weighted by the sum over spaces of its per-space L2 decoder norms ("L1-of-norms"). The alternative weight $\sqrt{\sum_{l} \lVert W^l_{dec,i}\rVert^2}$ ("L2-of-norms") is not used. A "fixed L1 coefficient" multiplies the penalty; its value is not stated.

## Model-diffing recipe
1. Pick one layer and run base and fine-tuned model on the same text. The source uses the middle-layer residual stream. One datapoint is the pair of residual vectors at one token position (pairing detail is inference).
2. Normalize each space separately so that each contributes comparably to the loss.
3. Train one crosscoder with $L = \{\text{base}, \text{finetuned}\}$ and the L1-of-norms penalty. The source uses 1 million features; tokens and optimizer settings are not stated.
4. For each feature compute the relative decoder norm. The source gives only the axis (0 = base only, 0.5 = shared, 1 = finetuned only); (inference) $r_i = \lVert W^{ft}_{dec,i}\rVert / (\lVert W^{base}_{dec,i}\rVert + \lVert W^{ft}_{dec,i}\rVert)$.
5. Histogram $r_i$ and look for three clusters. The source finds four to five thousand model-specific features per model out of 1 million.
6. For shared features compute the cosine similarity of the two decoder vectors. Low or negative values may mark a base concept used "in a new way" (authors' suspicion, untested).
7. Interpret exclusive features from their activating examples; the authors found it more productive to inspect features active on tokens of interest.

## Pitfalls stated in the source
- The L2-of-norms penalty "results in uncovering only shared features".
- "The majority of the model-exclusive features are not immediately interpretable"; the showcased ones are cherrypicked.
- About 2x more training FLOPs than per-layer SAEs for the same eval loss.
- The causal picture implied by a crosscoder "may differ from the mechanism actually used by the model"; be "very careful" when using it to inform [[Activation patching]] or other interventions.
- The reconstruction error "may be quite significant" and is hard to interpret.
- Crosscoders may "exploit correlations to work on distribution" without generalizing; mechanistic faithfulness is not formalized.
- Strictly causal variants "cannot capture the computation performed by attention layers".
- The authors only suspect crosscoder diffing is preferable "when model finetuning involves a greater fraction of compute relative to the compute used for pretraining"; applying a base-model SAE to the fine-tune is a competing kind of diff.

## Related
[[Tan et al. 2026 - Efficient Transcoder Adaptation]] compares a base model and its fine-tune with adapted per-layer transcoders and [[Attribution graphs]] instead of a shared dictionary.
