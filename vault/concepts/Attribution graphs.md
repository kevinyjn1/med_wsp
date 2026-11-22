---
type: concept
tags: [concept, attribution-graphs, circuits, transcoders]
---

# Attribution graphs

A per-prompt causal graph of how a model produced one output token, computed on a "local replacement model" in which feature-to-feature effects are exactly linear. Everything below is from [[Ameisen et al. 2025 - Circuit Tracing]] unless tagged "(inference)".

## Pipeline
1. **Dictionary.** Train a sparse replacement for the MLPs. The source uses a cross-layer transcoder (see [[Transcoders]]): $\mathbf{a}^{\ell} = \text{JumpReLU}(W_{enc}^{\ell}\mathbf{x}^{\ell})$, $\hat{\mathbf{y}}^{\ell} = \sum_{\ell' \le \ell} W_{dec}^{\ell' \to \ell}\mathbf{a}^{\ell'}$, trained with $\sum_{\ell}\lVert\hat{\mathbf{y}}^{\ell}-\mathbf{y}^{\ell}\rVert^2 + \lambda\sum_{\ell,i}\tanh(c\,\lVert\mathbf{W}_{dec,i}^{\ell}\rVert\,a_i^{\ell})$.
2. **Local replacement model** for prompt $p$: swap the MLPs for the dictionary, freeze attention patterns and normalization denominators at their values on $p$, and add an error term per (position, layer) equal to true MLP output minus reconstruction. Activations and logits then match the real model exactly.
3. **Nodes.** Active features per token position, token embeddings, error nodes, and output tokens (95% of probability mass, at most 10).
4. **Edges.** $A_{s \to t} = a_s w_{s \to t} = a_s \sum_{\ell_s \le \ell < \ell_t} (W_{dec,s}^{\ell_s \to \ell})^{T} J^{\blacktriangledown}_{c_s,\ell \to c_t,\ell_t} W_{enc,t}^{\ell_t}$, where $J^{\blacktriangledown}$ is the model's Jacobian with stop-gradients on MLP outputs, attention patterns and normalization denominators. In practice, for each target node: inject its input vector (encoder vector, or the gradient of logit minus mean logit), run one backward pass, and dot the gradient with each source's output vector (decoder, embedding or error vector). A feature's preactivation equals the sum of its incoming edges.
5. **Prune.** Normalize absolute input edges per node to sum to 1 ($A$), compute indirect influence $B = (I-A)^{-1} - I$, score nodes by probability-weighted influence on the logit nodes, keep nodes covering 0.8 of the influence and edges covering 0.98.
6. **Interpret.** Label features from dataset examples; group related nodes into supernodes by hand.
7. **Validate.** Steer features in the real model (constrained patching, see [[Activation patching]]) and check that downstream features and logits move as predicted. Report the replacement and completeness scores (share of influence not flowing through error nodes).
8. **Optional global view.** Virtual weights $V_{st} = \langle \sum_{\ell \in L_{st}} W_{dec}^{s,\ell}, W_{enc}^{t} \rangle$, reweighted by coactivation as $V_{ij}^{\text{TWERA}} = \frac{\mathbb{E}[a_j a_i]}{\mathbb{E}[a_j]} V_{ij}$ to suppress interference.

## What a graph does not show
- Why attention attends where it does (QK circuits). The source's multiple-choice example is a failure case.
- Anything inside error nodes, and inactive (suppressed) features.
- Ground truth: a graph is a hypothesis about the replacement model until interventions confirm it.

## Low-budget approximation, per the source
- The source states that "alternatives like per-layer transcoders or even MLP neurons can be used instead" of a CLT, keeping the remaining steps, "and still produce useful insights". With per-layer transcoders the decoder sum in step 4 has a single term.
- Stated price: on the 18L examples per-layer transcoders gave longer paths (3.7 vs 2.3) and a lower replacement score (0.37 vs 0.61 at 10M features), with chains of near-duplicate features across layers. Neurons need no training but are less interpretable.
- Adaptive generation: start at the logit nodes and attribute back only from the most influential nodes found so far.
- Speculative idea in the source: learn layer-to-layer linear maps shared by all features instead of per-feature cross-layer decoders.
- (inference) For med_wsp: per-layer transcoders on the few layers flagged by patching, error nodes for all other MLPs, adaptive graphs from a correct-minus-incorrect logit-difference node, and replacement / completeness scores reported for every graph.

Related: [[Crosscoders]], [[Lindsey et al. 2024 - Sparse Crosscoders]], [[Tan et al. 2026 - Efficient Transcoder Adaptation]].
