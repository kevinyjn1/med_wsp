---
type: concept
title: "Transcoders"
tags: [concept, transcoders, dictionary-learning, circuits, fine-tuning]
---

# Transcoders

Scope: written only from [[Tan et al. 2026 - Efficient Transcoder Adaptation]] and [[Lindsey et al. 2024 - Sparse Crosscoders]]. "(inference)" marks my own reading.

## What it is, versus an SAE
A sparse autoencoder (SAE) "reads and predicts the same layer": it rebuilds one activation vector from a wide sparse code. A transcoder "reads from one layer and predicts the next" (Lindsey et al.). In the form used by Tan et al. it reads the input of an MLP block and predicts that block's output, so it can be swapped into the model in place of the MLP. An SAE describes what is represented at one point; a transcoder describes what a layer computes. Lindsey et al. treat both as special cases of [[Crosscoders]].

## Training objective
Architecture as written by Tan et al.:

$\mathrm{TC}(x) = W_1\,\mathrm{ReLU}(W_2 x + b_2) + b_1$

Loss (their Eq. 1), one transcoder per MLP layer:

$L_{TC} = \lVert \mathrm{MLP}(x) - \mathrm{TC}(x) \rVert_2^2 + \lambda_1 \lVert \theta_{TC}(x) \rVert_1$

Here $x$ is the MLP input and $\theta_{TC}(x)$ the sparse feature activations; (inference) $\theta_{TC}(x) = \mathrm{ReLU}(W_2 x + b_2)$. The first term matches the MLP output, the second enforces sparsity. Tan et al. use 16,384 features per layer for Gemma2-2b ("2K" hidden size, 26 layers). $\lambda_1$ is not stated. Their quoted cost of training from scratch is about 1B tokens and 210 H100 hours.

## Use in circuit analysis
1. Replace every MLP by its transcoder and keep attention unchanged; this is the replacement model.
2. Measure how well it mimics the real model. Tan et al. plot Top-1 error rate and KL divergence against L0.
3. On a prompt, compute feature-to-feature attributions and draw [[Attribution graphs]] (method of [[Ameisen et al. 2025 - Circuit Tracing]]); prune, then merge similar features into supernodes.
4. Label features from their activating text and from direct logit attribution through the unembedding.
5. Weight-level reading (Lindsey et al., for a transcoder-like masked crosscoder): the input from an upstream feature to a downstream feature is the dot product of the downstream encoder vector with the upstream decoder vector, weighted by the upstream activation.

## Adapting to a fine-tuned model (PAT, Tan et al.)
Preconditions: base-model transcoders $TC_1$ exist; the fine-tune changed only MLP weights with attention frozen (their FTFA); the fine-tuning data $D$ and loss are available.
1. Stage 1: start from $TC_1$ and train the transcoders on $D$ with the fine-tuning loss:
   $\theta' = \arg\min_{\theta_{TC}} \mathbb{E}_{x\sim D}\left[L(f_{\theta_{TC}}(x), y)\right], \quad \theta_{TC}^{(0)} = \theta_{TC_1}$
   Neural Tangent Kernel argument: MLP and transcoder share architecture and activation and start with almost equal outputs, so the same data and loss move them the same way, giving $TC_2' \approx \mathrm{MLP}_2$ in the infinite-width limit.
2. Stage 2: refine $TC_2'$ with Eq. 1 against the fine-tuned MLP on a sampled subset $D_{sample}$.

Reported cost: 1M tokens and 0.2 H100 hours for 26 layers of 16,384 features.

## Caveats from the sources
- Base transcoders used unchanged on a fine-tuned model gave "nearly zero Top-1 accuracy" as a replacement model (Tan et al.).
- The adapted 16k transcoders still show Top-1 error near 0.885 (my reading of their plot), similar to the 16k baseline.
- PAT is not tested when attention weights changed, and stage 1 is not ablated against stage 2.
- Transcoders explain MLP computation only. Lindsey et al. note that strictly causal coders "cannot capture the computation performed by attention layers"; head-level questions need other tools such as [[Activation patching]].
