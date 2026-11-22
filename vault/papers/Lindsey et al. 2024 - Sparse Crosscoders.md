---
type: paper
title: "Sparse Crosscoders for Cross-Layer Features and Model Diffing"
short: "Lindsey et al., 2024"
authors: [Jack Lindsey, Adly Templeton, Jonathan Marcus, Thomas Conerly, Joshua Batson, Christopher Olah]
year: 2024
venue: "Transformer Circuits Thread (Anthropic research update), published Oct 25, 2024"
url: "https://transformer-circuits.pub/2024/crosscoders/index.html"
local_text: "papers/pdf/text/crosscoders_2024.txt"
tags: [paper, crosscoders, model-diffing, sparse-autoencoders, dictionary-learning, fine-tuning]
read_status: full-text-read
---

# Lindsey et al., 2024 - Sparse Crosscoders

> Reading conventions: plain statements paraphrase or quote the source. Anything marked "(inference)" is mine. "Not stated" means I searched the text and the HTML and did not find it. The source calls itself a "preliminary note" and a "research update", "not at the level of quality or rigor we hold our full papers to". All figures are images in the HTML; values marked "read from figure" are my approximate readings or annotations printed inside the figure, not in the text file.

## TL;DR
A crosscoder is a sparse dictionary whose features read from and write to several activation spaces at once: several layers of one model, or the same layer of several models. Sparse autoencoders and [[Transcoders]] are treated as special cases. The loss is summed reconstruction error over all spaces plus an L1 penalty on each feature activation weighted by the sum of that feature's per-space decoder norms. On an 18-layer model a global acausal crosscoder beats 18 per-layer SAEs at equal total feature count but is about 2x less FLOP efficient. For model diffing, a 1 million feature crosscoder on the middle-layer residual stream of Claude 3 Sonnet and its pre-finetuning base model splits features into three clusters by relative decoder norm: shared features, and "between four and five thousand model-specific features for each model". The authors report that "the majority of the model-exclusive features are not immediately interpretable".

## Problem
Three motivations in the source's framing: (1) cross-layer superposition, where a feature is computed jointly by several layers; (2) persistent features that stay in the residual stream for many layers, which per-layer SAEs duplicate and which make circuits look more complex than they are; (3) model diffing: finetuning "may induce a small set of changes in a model, amidst a background of mostly unchanged structure. We would like methods that can isolate and interpret these changes."

## Method (step by step)

1. **Encoder.** For datapoint $x_j$ (one token position) with activations $a^l(x_j)$ in each space $l \in L$:
   $f(x_j) = \mathrm{ReLU}\left(\sum_{l\in L} W^l_{enc}\, a^l(x_j) + b_{enc}\right)$
   One shared feature vector $f$, one encoder matrix per space, one shared encoder bias.
2. **Decoder.** Each space is reconstructed from the same features with its own decoder matrix and bias:
   $a^{l'}(x_j) = W^l_{dec}\, f(x_j) + b^l_{dec}$
3. **Loss.**
   $L = \sum_{l\in L} \lVert a^{l}(x_j) - a^{l'}(x_j) \rVert^2 + \sum_{l\in L}\sum_i f_i(x_j)\, \lVert W^l_{dec,i} \rVert$
   The sparsity term can be rewritten as
   $\sum_i f_i(x_j)\left(\sum_{l\in L} \lVert W^l_{dec,i} \rVert\right)$
   i.e. the L1 penalty on feature $i$ is weighted by the L1 norm (over spaces) of the per-space L2 decoder norms $\lVert W^l_{dec,i} \rVert$. The formula is printed without a sparsity coefficient; Section 3.1 says "We used a fixed L1 coefficient for the sparsity penalty" (value not stated). (inference) Implement the second term as $\lambda \sum_i f_i \sum_l \lVert W^l_{dec,i}\rVert$.
4. **Why L1-of-norms and not L2-of-norms.** The alternative weight is $\sqrt{\sum_{l\in L} \lVert W^l_{dec,i} \rVert^2}$ (treat all decoder weights of a feature as one vector). Reasons given for the L1 version: (a) the loss is directly comparable to the sum of per-layer SAE losses, whereas the L2 version gives a "loss bonus" for spreading features across layers; (b) the L2 version encourages features to spread out, while the L1 version gives no such incentive, and "in the context of model diffing ... it uncovers a mix of shared and model-specific features, while the L2-of-norms version results in uncovering only shared features". The L2 version "more efficiently optimizes the frontier of MSE and global L0". "In this report, all experiments used the L1-of-norms version."
5. **Activation normalization.** "We separately normalize the activations of each layer prior to training the crosscoder, so that each layer contributes comparably to the loss." Exact normalization: not stated.
6. **Variants (table image in Section 2.1).** Cross-layer approach: per-layer SAEs / shared SAE / sparse crosscoder. Causality: acausal (reads and predicts past, present and future layers; "makes circuit analysis very difficult") / weakly causal (reads one layer, predicts that layer and later ones) / strictly causal (reads one layer, predicts only later activations; "the cross-layer generalization of a transcoder"). Locality: global / local "conv" window / skip layer / cross model ("Crosscoders may interact with layers from different models to create canonical feature sets between them"). Target: residual stream / layer outputs.
7. **Model-diffing setup (Section 4.3).** Use the same architecture with the index $l$ running over models instead of layers: one crosscoder "with 1 million features on the residual stream activations from the middle layer of Claude 3 Sonnet and the base model from which it was finetuned". (inference) Each training datapoint is the pair of residual vectors at the same token position of the same text in the two models; the source states that crosscoders are "trained using only a single token position's activations at a time".
8. **Classify features by relative decoder norm.** "We looked at the relative norms of feature decoder weights in the two models." The figure axis "Relative Decoder Norm Strength" runs from 0.0 "(base only)" through 0.5 "(shared)" to 1.0 "(finetuned only)". The formula is not given. (inference) The axis is consistent with $r_i = \lVert W^{ft}_{dec,i}\rVert \,/\, \left(\lVert W^{base}_{dec,i}\rVert + \lVert W^{ft}_{dec,i}\rVert\right)$. Thresholds separating the three clusters: not stated.
9. **Check alignment of shared features.** For shared features, compute the cosine similarity between the feature's decoder vector in the two models.
10. **Related masked variant (Section 3.3.3, pre/post-MLP crosscoder).** Encoders read only the pre-MLP residual stream; decoders reconstruct both the pre-MLP activity and the MLP output. Relative decoder norms are trimodal: pre-only, shared, post-only ("newly computed"). Inputs to a newly computed feature are obtained "by taking the dot product of the downstream feature's encoder vector with the upstream features' decoder vectors, and weighting by the source features activation". "Stable" features are defined "arbitrarily" as relative post-MLP decoder norm between 0.3 and 0.7.

Not stated anywhere in the source: training data distribution, number of training tokens or steps, optimizer, learning rate, batch size, L1 coefficient, initialization, dead-feature handling, decoder-norm constraints, model sizes and residual widths, and any code or checkpoints.

## Experiments and numbers

**Cross-layer (Section 3.1).** Global acausal crosscoder on the residual stream of all layers of "an 18-layer model" versus 18 per-layer SAEs, fixed L1 coefficient, sweep over training steps and feature count.
- FLOPs accounting: "a global, acausal crosscoder with F total features uses the same number of training FLOPS as a collection of per-layer SAEs with F features each (and thus with L×F total features)".
- Eval loss (MSE + decoder-norm-weighted L1, summed over layers): "controlling for the total number of features across layers, crosscoders substantially outperform per-layer SAEs"; figure annotation: "Crosscoders are ~8x more feature efficient". Feature counts on the axis run from about $2^{12}$ to $2^{22}$ (read from figure).
- "With respect to training FLOPS, crosscoders are less efficient than per-layer SAEs at achieving the same eval loss, by a factor of about 2 at large compute budgets."
- (MSE, total L0) at FLOP-equivalence, figure annotation: "FLOP equivalent crosscoders achieve 15-40x lower total L0 than SAEs, with better MSE". Crosscoder total L0 is below about 50 while per-layer SAE total L0 is about 700 to 1460, at normalized reconstruction error about 0.16 to 0.33 (read from figure).

**Feature analysis (Section 3.2).** Decoder norms of 50 random features over layers: "most features tend to peak in strength in a particular layer, and decay in earlier and later layers". Decoder directions are "much more stable across layers than would be expected by chance" but "drift substantially across layers".

**Masked variants (Section 3.3).** Local windows of $K$ layers: eval loss "interpolated fairly linearly" from $K=1$ to $K=n_{layers}$, no inflection point. Weakly causal: FLOPs efficiency between per-layer SAEs (slightly worse) and global acausal (slightly better); by dictionary size it "lagged behind that of global crosscoders by a factor of 3 to 4". Strictly causal "cross-layer transcoders" (read residual at layer L, predict MLP outputs at L, L+1, ...): preliminary, a mix of local and global features. Pre/post-MLP: trimodal relative norms (read from figure: spikes of about 7.7k features at fully pre-MLP and about 12k at fully post-MLP, broad hump at equal magnitude); cosine similarity of stable features between pre and post space is positive on average, high variance, "relatively low in absolute terms" (read from figure: histogram centred near 0.25, range about -0.6 to 0.8).

**Model diffing, Sonnet finetuning (Section 4.3).**

| Quantity | Value |
|---|---|
| Models | Claude 3 Sonnet and "the base model from which it was finetuned" |
| Activations | residual stream, middle layer, both models |
| Dictionary size | 1 million features |
| Model-specific features | "between four and five thousand ... for each model" |
| Training tokens, L1 coefficient, residual width | not stated |

- Finetuned-only examples: a refusal feature on dangerous requests; a code review feature; a feature for humans asking the Assistant personal questions about itself. Figure annotation adds "Human questioning/correcting Assistant, Step-by-Step Reasoning in Math".
- Base-only examples: a feature for LLMs "roleplaying" or being cast as a character in a system prompt; a feature for dialogues between humans and a smartphone assistant.
- "These features are cherrypicked. Unfortunately, we've also found that the majority of the model-exclusive features are not immediately interpretable. However, inspecting the features that activate on tokens of interest often reveals clearly interpretable features."
- Shared features: decoder vectors "in almost all cases ... highly aligned"; "for a few thousand features, the correlation was very low or even negative", not investigated, suspected to be cases where the finetuned model uses a base-model concept "in a new way".
- Figure reading (inference): the shared peak is narrow, roughly 0.4 to 0.6 on the relative-norm axis. The tallest histogram bar is about 31K and the bars sum to roughly 100K, and the cosine histogram's largest bin is about $10^4$, so both plots seem to show a subset of the 1 million features. The source does not say so.

**Model diffing over (layer, scale) (Section 4.4).** One acausal crosscoder on ten evenly spaced layers from three models of increasing size; per-feature 30-dimensional vector of decoder norms; nonnegative matrix factorization with four components. Components: early layers of all models, late layers of all models, middle layers of the smallest model, middle layers of the two larger models. The feature axis in the figure ends near 75k (read from figure).

## What it establishes
- A single dictionary with per-space encoders and decoders and a decoder-norm-weighted L1 penalty can be trained across layers and across models.
- On one 18-layer model, crosscoders use dictionary features more efficiently than per-layer SAEs and training FLOPs less efficiently (about 2x).
- On one base/finetune pair at one layer, the relative decoder norm is clearly trimodal, and a small fraction of features (4 to 5 thousand per model out of 1 million) is model-specific.
- Some model-specific features are interpretable and match what the finetune is known to add (refusal, assistant persona).
- The choice of L1-of-norms matters for diffing: the authors report that the L2-of-norms version finds only shared features.

## What it does NOT establish
- That model-specific features are causally responsible for behavioural differences. No ablation, steering or patching of exclusive features is reported.
- That exclusive features are mostly interpretable. The authors say the majority are not.
- Any result for small open models, domain fine-tunes (for example medical), or more than one layer of a base/finetune pair.
- A reproducible training recipe. Hyperparameters, data and token counts are absent and no code is released in the source.
- That a crosscoder is the better diffing tool for light fine-tunes. The authors only "suspect" it is preferable "when model finetuning involves a greater fraction of compute relative to the compute used for pretraining".
- Quantitative interpretability: "we do not conduct a systematic analysis of qualitative feature interpretability in this work".
- How features map to weights, heads or MLPs. The diff is in activation space at one layer.

## Limitations stated by the authors
- Preliminary research update, not at full-paper rigor; model-diffing results "very preliminary" and "a bit mixed".
- "This strategy produces many features we don't understand"; most model-exclusive features are not immediately interpretable.
- Low or negative decoder-alignment features were not investigated.
- Crosscoders cost about 2x more training FLOPs than per-layer SAEs for the same eval loss.
- Causal description may differ from the underlying model: crosscoders are "incentivized to represent all within-token-position cross-layer correlations as cross-layer features", so "we need to be very careful in using crosscoders to inform interventional experiments such as patching".
- Crosscoder errors "may be important and extremely difficult to interpret".
- "Mechanistic faithfulness" is not formalized; crosscoders may exploit on-distribution correlations that do not generalize.
- Strictly causal crosscoder layers "cannot capture the computation performed by attention layers".
- Open comparison with the alternative of applying a base-model SAE to the finetuned model (Kissane et al., who found SAEs "often transfer well"); "More work is needed to determine the conditions under which each of these approaches provides a more natural description".

## Implementation notes for med_wsp
All sizing below is (inference); the source gives no compute numbers for the diffing experiment. Assumed Phi-3.5-mini residual width 3072 (general knowledge, verify in config.json), so a two-model one-layer crosscoder sees $2 \times 3072 = 6144$ input dimensions.

| Features F | Parameters (enc + dec) | fp32 weights | With Adam state (3 copies) | Fits in 36 GB |
|---|---|---|---|---|
| 16,384 | 201M | 0.8 GB | 2.4 GB | yes |
| 65,536 | 805M | 3.2 GB | 9.7 GB | yes |
| 131,072 | 1.61B | 6.4 GB | 19.3 GB | tight next to two models |
| 1,048,576 (the source's size) | 12.9B | 51.5 GB | 155 GB | no |

- Running both 3.8B-parameter models in fp16 is about 7.6 GB each, 15.2 GB together, leaving room for a dictionary of up to about 65k features (inference).
- Paired activations cost 6144 values per token: about 12.3 KB in fp16. 1M tokens is about 12.3 GB, 10M tokens about 123 GB. With limited disk, harvest activations on the fly or keep a small shuffled buffer (inference).
- By the source's FLOPs rule, a two-space crosscoder with F features costs about as much as two SAEs with F features each (inference by analogy with the L-layer statement).
- Minimum faithful recipe from the source: ReLU encoder summing per-model encoders, per-model decoders, L1-of-norms penalty, per-space activation normalization, residual stream at one middle layer. Everything else must be chosen by us.
- Gotcha: with only 16k to 65k features, the number of model-exclusive features may be small. In the source they were about 0.4 to 0.5 percent of 1M per model; whether that fraction carries over to a small dictionary and a light fine-tune is unknown (inference).
- Gotcha: the authors suspect crosscoder diffing suits heavier fine-tunes. If MediPhi is a light change relative to pretraining (see [[Corbeil et al. 2025 - MediPhi]]), the base-SAE-transfer style of diff is a competing baseline (inference).

## Relevance to med_wsp (provisional)
- Provisional idea: a middle-layer Phi-3.5 / MediPhi crosscoder gives a list of MediPhi-only and Phi-only features; intersect their activations with MedQA items MediPhi gets right and Phi gets wrong to rank candidate features for Astra to inspect.
- Provisional idea: unlike the adaptation route in [[Tan et al. 2026 - Efficient Transcoder Adaptation]], a crosscoder needs no pre-trained base dictionary and no assumption about which weights changed, so it applies to a merged fine-tune.
- Provisional caution: the source warns against naive use of crosscoders for patching. Any exclusive feature we pick should be checked with [[Activation patching]] on the real models before it guides a weight edit ([[Weight-space programming]]).
- Provisional caution: a residual-stream diff at one layer says where representations differ, not which heads or MLPs wrote the difference; it complements, and does not replace, weight diffs and [[Transcoders]].

## Open questions
- What exact formula and thresholds define "model-specific"? Not given.
- How many tokens and what data mix are needed for a stable three-cluster result on a 3.8B pair?
- Does the trimodal structure appear at 16k to 65k features, or do exclusive features only emerge at large dictionary sizes?
- Are exclusive features genuine new directions, or artifacts of the sparsity penalty? The source does not test this.
- Which layer should we diff? The source used "the middle layer" without justification.
- For a light medical fine-tune, is base-SAE transfer (Kissane et al., as cited by the source) more informative than a crosscoder?

## Ledger card
**Full citation:** Lindsey, J.; Templeton, A.; Marcus, J.; Conerly, T.; Batson, J.; Olah, C. 2024. Sparse Crosscoders for Cross-Layer Features and Model Diffing. Transformer Circuits Thread, Anthropic, Oct 25, 2024.
**Source / DOI / URL:** https://transformer-circuits.pub/2024/crosscoders/index.html ; no DOI; local HTML `papers/pdf/crosscoders_2024.html`.
**Why I read it:** defines the crosscoder loss and the relative-decoder-norm method for diffing a base model against its fine-tune.
**Problem addressed:** cross-layer superposition, duplicated persistent features, and isolating what finetuning changes.
**Method / representation:** sparse dictionary with per-space encoders and decoders, shared ReLU features, MSE plus L1 weighted by summed per-space decoder norms; relative decoder norm and decoder cosine similarity for diffing.
**Assumptions:** linear features; paired activations at matching token positions; per-space normalization; an L1 relaxation of sparsity over (feature, space) pairs.
**Evidence used by the paper:** experiments: 18-layer crosscoder versus per-layer SAEs, masked variants, 1M-feature Sonnet base/finetune diff, three-model ten-layer NMF / analysis: decoder norm and direction statistics, cherrypicked feature examples / datasets or systems: an 18-layer model, Claude 3 Sonnet and its base, three models of increasing size (all internal, data not stated).
**Safe citation claims:** "Lindsey et al. introduce sparse crosscoders and apply them to diff Claude 3 Sonnet against its base model at the middle layer with 1 million features"; "They report a trimodal relative decoder norm with four to five thousand model-specific features per model"; "They report that most model-exclusive features were not immediately interpretable".
**Unsafe / unsupported citation uses:** "crosscoders reliably find the features responsible for fine-tuning gains"; "model-exclusive features are causally validated"; "crosscoders are safe to use for patching experiments"; any specific training hyperparameters; any claim about small open models.
**Relation to our project:** overlapping method
**Verification status:** full text read: yes; key claim checked: yes (formulas checked against the HTML math source; figure annotations read from the embedded images)

## Links
- Concepts: [[Crosscoders]], [[Transcoders]], [[Activation patching]], [[Attribution graphs]], [[Weight-space programming]]
- Papers: [[Tan et al. 2026 - Efficient Transcoder Adaptation]] (alternative route to comparing base and fine-tune), [[Ameisen et al. 2025 - Circuit Tracing]] (later work using cross-layer transcoders, the strictly causal variant named here), [[Corbeil et al. 2025 - MediPhi]] (our fine-tuned model)
