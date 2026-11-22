---
type: concept
tags: [concept, activation-patching, interventions, causal-validation]
---

# Activation patching

A family of causal interventions: overwrite an internal activation during a forward pass (with a value from another run, or a modified value) and measure the change in output.

## What [[Ameisen et al. 2025 - Circuit Tracing]] says
- Related work: activation patching, causal mediation analysis, path patching (the extracted text reads "patch patching") and distributed alignment search are the common tools for "refining hypotheses and isolating causal pathways", but "generally do not provide interpretable (i.e. linear) edge weights between units". Patching is "computationally expensive, requiring a forward pass per step", which motivated gradient-based attribution patching.
- The paper's own interventions act on transcoder features and serve to validate [[Attribution graphs]]:
  - Constrained patching: change a feature's decoding over a chosen layer range (MLP outputs clamped to precomputed values), then run forward from the end of the range. No knock-on effects inside the range.
  - Iterative patching: also recompute features inside the range. Captures knock-on effects, but errors cascade.
  - Steering: multiply a feature's activation by $M$. Inhibition typically uses $M=-1$ or stronger instead of zero-ablation; strengths are ad hoc.
  - Ablation is used for evaluation: graph influence vs ablation effect on downstream features has Spearman 0.72.
  - Attention patterns stay frozen, so QK-mediated effects are excluded.
- Caveats: direct edges are nearly guaranteed to be confirmed under frozen attention, so the informative test is multi-step effects; adding decoder vectors at every layer double-counts; faithfulness metrics are sensitive to ablation methodology (the source cites Miller et al.).

## Standard background (background, not from the source)
- Cache activations from a clean run and patch them into a corrupted run ("denoising", tests sufficiency), or the reverse ("noising", tests necessity).
- Sites: residual stream, attention-head output, MLP output, per layer and token position.
- Usual metric: normalized logit difference, $\frac{LD_{patched} - LD_{corrupt}}{LD_{clean} - LD_{corrupt}}$.

## Cross-model patching, base vs fine-tune (our provisional design, not from the source)
1. Select MedQA items that MediPhi ([[Corbeil et al. 2025 - MediPhi]]) answers correctly and Phi-3.5 answers incorrectly; use identical prompts and tokenization.
2. Run both models and cache MediPhi activations at every (layer, component, position).
3. Baselines: $LD = \text{logit}(\text{correct letter}) - \text{logit}(\text{incorrect letter})$ for each model.
4. For each site, rerun Phi-3.5 with that activation replaced by MediPhi's and record $\Delta LD = LD_{patched} - LD_{base}$, optionally divided by $LD_{MediPhi} - LD_{base}$.
5. Rank sites by mean $\Delta LD$ across items. Run the reverse direction (Phi-3.5 into MediPhi) for necessity.
6. Controls: patch MediPhi activations from a different question; check that top sites recur across items.

Caveats (inference): this assumes the two models share a residual-stream basis, plausible for a fine-tune but untested here. Because the source shows multiple-choice answers can hinge on attention, attention-head outputs and patterns should be among the patched sites.

Related: [[Transcoders]], [[Weight-space programming]].
