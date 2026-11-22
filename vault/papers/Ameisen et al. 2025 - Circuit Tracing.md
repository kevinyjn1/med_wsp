---
type: paper
title: "Circuit Tracing: Revealing Computational Graphs in Language Models"
short: "Ameisen et al., 2025"
authors: ["Emmanuel Ameisen", "Jack Lindsey", "Adam Pearce", "Wes Gurnee", "Nicholas L. Turner", "Brian Chen", "Craig Citro", "David Abrahams", "Shan Carter", "Basil Hosmer", "Jonathan Marcus", "Michael Sklar", "Adly Templeton", "Trenton Bricken", "Callum McDougall", "Hoagy Cunningham", "Thomas Henighan", "Adam Jermyn", "Andy Jones", "Andrew Persic", "Zhenyi Qi", "T. Ben Thompson", "Sam Zimmerman", "Kelley Rivoire", "Thomas Conerly", "Chris Olah", "Joshua Batson"]
year: 2025
venue: "Transformer Circuits Thread (Anthropic), published March 27, 2025"
url: "https://transformer-circuits.pub/2025/attribution-graphs/methods.html"
local_text: "papers/pdf/text/circuit_tracing_methods_2025.txt"
tags: [paper, attribution-graphs, transcoders, cross-layer-transcoder, circuits, interventions, interpretability]
read_status: full-text-read
---

# Ameisen et al., 2025 - Circuit Tracing

Conventions in this note: everything is from the source unless tagged "(inference)" (my own reasoning) or "(background, not from the source)". "Not stated" means the extracted text gives no value (several results exist only as figures, which the text extraction does not contain).

## TL;DR
The paper describes a method for producing per-prompt "attribution graphs": causal graphs over sparse features, token embeddings, error terms and output logits. The MLPs of the model are replaced by a cross-layer transcoder (CLT); attention patterns and normalization denominators are frozen and per-(position, layer) error terms are added, so that the "local replacement model" reproduces the real activations exactly and feature-to-feature effects are linear. Graphs are pruned (default: about 10x fewer nodes for about 20% less explained behavior), features are hand-grouped into "supernodes", and hypotheses are validated by steering features in the underlying model ("constrained patching"). On the 18-layer research model ("18L") the largest CLT (10M features) matches the model's top-1 next token on 50% of filtered prompts, with normalized MSE of about 11.5% and L0 of 88; on Claude 3.5 Haiku the largest CLT (30M features) reaches 21.7% normalized MSE and L0 of 235. Graph influence predicts ablation effects (Spearman 0.72 for feature-to-feature), but faithfulness of the replacement model decays over layers and the method does not explain attention patterns (it fails on a multiple-choice example).

## Problem
Neurons are polysemantic (superposition), and gross components (whole MLP blocks, attention heads) "play a multitude of unrelated roles across the data distribution". Sparse-coding features (SAEs, transcoders, crosscoders) give interpretable units, but the design space for building circuits out of them is large. The paper states its current approach: which dictionary to use, how to define linear, well-defined feature-feature attributions on a single prompt, how to prune and visualize the result, and how to validate it, with stated limitations. It is the methods paper for the companion "On the Biology of a Large Language Model" (Claude 3.5 Haiku case studies).

## Method (step by step)

### 1. Cross-layer transcoder (CLT) architecture
- The CLT has features divided into $L$ layers (same number as the model). A layer-$\ell$ feature reads the residual stream at layer $\ell$ (at the MLP input: "upon reaching the input to the MLP in layer $\ell$, we compute the activations of the cross-layer transcoder features whose encoders live in layer $\ell$") and writes to the MLP-output reconstruction of layers $\ell, \ell+1, \ldots, L$, with a separate decoder vector per target layer. All features in all layers are trained jointly.
- Encoder: $\mathbf{a}^{\ell} = \text{JumpReLU}\left(W_{enc}^{\ell}\mathbf{x}^{\ell}\right)$, where $\mathbf{x}^{\ell}$ is the residual stream at layer $\ell$.
- Decoder (reconstruction of the layer-$\ell$ MLP output $\mathbf{y}^{\ell}$): $\hat{\mathbf{y}}^{\ell} = \sum_{\ell'=1}^{\ell} W_{dec}^{\ell' \to \ell}\,\mathbf{a}^{\ell'}$.
- No bias terms appear in the source formulas; whether biases exist is not stated.
- JumpReLU definition: the source does not write it out (it cites Rajamanoharan et al.). Standard form: $\text{JumpReLU}_{\theta}(z) = z \cdot H(z - \theta)$ with a learned per-feature threshold $\theta$ and Heaviside step $H$ (background, not from the source).
- Parameter count: a CLT has the same number of encoder parameters as per-layer transcoders (PLTs) with the same feature count, but "approximately n_layers/2 times as many decoder parameters".

### 2. CLT loss and training
- Reconstruction loss: $L_{\text{MSE}} = \sum_{\ell=1}^{L} \lVert \hat{\mathbf{y}}^{\ell} - \mathbf{y}^{\ell} \rVert^2$.
- Sparsity penalty: $L_{\text{sparsity}} = \lambda \sum_{\ell=1}^{L}\sum_{i=1}^{N} \tanh\left(c \cdot \lVert \mathbf{W}_{dec,i}^{\ell} \rVert \cdot a_i^{\ell}\right)$, where $N$ is the number of features per layer, $\mathbf{W}_{dec,i}^{\ell}$ is the concatenation of all decoder vectors of feature $i$, and $\lambda$, $c$ are hyperparameters (values not stated).
- Pre-activation loss (Haiku runs only, to prevent dead features): $\sum_f \text{ReLU}(-h_f)$ with coefficient $3\times10^{-6}$, where $h_f$ is the pre-nonlinearity activation of feature $f$.
- Total loss: $L_{\text{MSE}} + L_{\text{sparsity}}$ (plus the pre-activation loss on Haiku).
- Training details that differ from the authors' earlier SAE setup:
  - Each residual stream layer and each MLP output is normalized separately before being used as input / target.
  - JumpReLU with a straight-through gradient estimator, bandwidth 1.0, initial threshold 0.03; gradient flows through the JumpReLU to all parameters, not only the threshold.
  - Sparsity coefficient ramps linearly from 0 to its final value over the entire training run.
  - Training steps scale "slightly sublinearly" with feature count; learning rate scales "approximately as one over the square root of the number of FLOPs" (absolute values, optimizer and batch size not stated).
  - 18L: constant sparsity penalty across sizes. Haiku: penalty increased with size. Target L0 "in the low hundreds".
  - Init: encoder $\sim U\left(-\tfrac{1}{\sqrt{n_{features}}}, \tfrac{1}{\sqrt{n_{features}}}\right)$; decoder $\sim U\left(-\tfrac{1}{\sqrt{n_{layers}\cdot d_{model}}}, \tfrac{1}{\sqrt{n_{layers}\cdot d_{model}}}\right)$.
  - Training data shuffled at the token level ("not shuffling leads to significantly worse performance").
  - Data: 18L CLTs trained on pretraining data only; Haiku CLT on a mix of pretraining and finetuning data.
  - CLTs need a comparable order of magnitude of tokens as per-layer dictionaries (rough estimate: benefit from about 2 times more tokens).
- Engineering: features are sharded across accelerators; per batch each accelerator computes partial predictions, an all-reduce combines them, then backward. Per accelerator, CLT and PLT use the same FLOPs but the CLT uses n_layers times more network bandwidth. Activations are stored to a distributed file system; both CLTs and all-layer PLTs need activations from every layer. Not-implemented optimizations: sparse kernels (could cut CLT FLOPs by up to n_layers/2), sharding the decoder over the output dimension.

### 3. Replacement model
Forward pass of the original model with two changes: (a) at the MLP input of layer $\ell$, compute the CLT features whose encoders live in layer $\ell$; (b) overwrite the layer-$\ell$ MLP output with the summed outputs of CLT features from this and previous layers, using their layer-$\ell$ decoders. Attention layers run normally (no freezing). The CLT therefore runs on off-distribution inputs (its own upstream reconstructions), and errors compound.

### 4. Local replacement model (for one fixed prompt $p$)
1. Substitute the CLT for the MLPs.
2. Use the attention patterns and normalization denominators recorded from the underlying model's forward pass on $p$ (frozen).
3. Add an error adjustment at every (token position, layer) equal to true MLP output minus CLT output on $p$.

After this, all activations and logits exactly match the underlying model on $p$. It can be viewed as one large fully connected network across tokens: inputs are the one-hot token vectors; neurons are the CLT features active at each position; weights ("virtual weights") are the summed linear paths between features through the residual stream and attention OV circuits, not passing through MLP/CLT layers; error terms act as bias-like nodes; the only nonlinearities are the feature activation functions. Exact agreement on $p$ does not guarantee the same mechanism (the authors compare it to a Taylor approximation at a point).

### 5. Attribution graph: nodes
- Output nodes: candidate output tokens, enough to cover 95% of probability mass, at most 10.
- Intermediate nodes: active CLT features at each token position, i.e. (feature, position) pairs.
- Input nodes: token embeddings.
- Error nodes: per (position, layer), the part of the MLP output not explained by the CLT. They have no input edges.

### 6. Attribution graph: edges
- Definition: $A_{s \to t} := a_s\, w_{s \to t}$, with $a_s$ the source activation and $w_{s\to t}$ the virtual weight, equivalently "the derivative of the preactivation of $t$ with respect to the source feature activation, with stop-gradients on all non-linearities in the local replacement model".
- $J^{\blacktriangledown}_{c_s,\ell_s \to c_t,\ell_t}$ is the Jacobian of the underlying model from the residual stream at (position $c_t$, layer $\ell_t$) back to the residual stream at (position $c_s$, layer $\ell_s$), with stop-gradient applied to MLP outputs, attention patterns and normalization denominators.
- Feature to feature (formula truncated in the extracted text; recovered from the local HTML):
  $A_{s \to t} = a_s\, w_{s \to t} = a_s \sum_{\ell_s \le \ell < \ell_t} \left(W_{dec,s}^{\ell_s \to \ell}\right)^{T} J^{\blacktriangledown}_{c_s,\ell \to c_t,\ell_t}\, W_{enc,t}^{\ell_t}$
- General form using node input/output vectors:
  - embedding or error source: $A_{s \to t} = v_{out,s}^{T}\, J^{\blacktriangledown}_{c_s,\ell_s \to c_t,\ell_t}\, v_{in,t}$
  - feature source: $A_{s \to t} = a_s \sum_{\ell_s \le \ell < \ell_t} \left(v_{out}^{\ell}\right)^{T} J^{\blacktriangledown}_{c_s,\ell \to c_t,\ell_t}\, v_{in,t}$
- Node vectors:
  - output node for token tok: $v_{in} = \nabla\left(\text{logit}_{tok} - \overline{\text{logit}}\right)$ (target logit minus mean logit);
  - feature node $s$: $v_{out}^{\ell} = W_{dec,s}^{\ell_s \to \ell}$ for $\ell_s \le \ell$, and $v_{in} = W_{enc,s}^{\ell_s}$;
  - embedding node: $v_{out} = \text{Emb}_{tok}$;
  - error node at (position $c$, layer $\ell$): $v_{out} = \text{MLP}_{\ell}(x_{c,\ell}) - \text{CLT}_{\ell}(x_c)$.
- Path expansion: $J^{\blacktriangledown}_{c_s,\ell_s \to c_t,\ell_t} = \sum_{p \in \mathcal{P}(c_s,c_t,\ell_s,\ell_t)} \pi_p$, with $\pi_p = \prod_i \pi_i$, where $\pi_i$ is the identity for a residual step and $a^{h_i}_{c_i \to c_{i+1}}\, OV_{h_i}$ for a step through attention head $h_i$.
- Linearity: the source prints $h_t = \sum_{\mathcal{S}_t} w_{s \to t}$ and says the preactivation of any feature node is "simply the sum of its incoming edges", with $\mathcal{S}_t$ the nodes at earlier layers and equal or earlier positions. (inference) The intended sum is over edge weights $A_{s\to t} = a_s w_{s\to t}$.
- Edges capture OV circuits (information moved by frozen attention) but not QK circuits (why attention moved it). Outgoing edges of a cross-layer feature aggregate its decodings at all layers it writes to.

### 7. Computing the edges in practice
Iterate over target nodes (output or feature nodes). For each:
1. Inject the target's input vector into the residual stream: for a CLT feature, its encoder vector at the layer and token position where it reads; for a logit node, the gradient of (logit minus mean logit) at the final-layer residual stream.
2. Run a backward pass through the underlying model with stop-gradients on MLP outputs and with normalization denominators and attention patterns fixed to their recorded values.
3. For a CLT source node: sum over layers of (decoder vector at that layer dotted with the gradient at that layer), times the feature's activation.
4. For an embedding or error source node: dot its vector with the gradient.

Cost is "linear in the number of active features in the prompt, and is dominated by the cost of the underlying model backwards pass". Adaptive variant: start from the logit nodes, keep a queue of feature nodes with the greatest logit influence, and compute input edges in queue order; for long prompts, repeatedly attribute back from the top-$k$ most influential explored nodes (unexplored nodes count as errors) until a target node count is reached.

### 8. Graph pruning
1. Take absolute values of the adjacency matrix and normalize the input edges of each node to sum to 1: $A$ (indexed target, source).
2. Indirect influence: $B = A + A^2 + A^3 + \cdots = (I - A)^{-1} - I$.
3. Logit influence score per node: average of the rows of $B$ belonging to logit nodes, weighted by token probability.
4. Node pruning: sort non-logit nodes by influence, keep the smallest prefix whose cumulative share of total influence reaches the threshold (default 0.8); pseudocode: `nodes_to_keep = cumulative_influence <= threshold`.
5. Recompute $A$ and the node scores on the pruned graph (logit nodes get their probability as score). Edge score = normalized edge weight times the logit influence of the edge's target node (`edge_score = A * node_score[:, None]`). Keep edges up to cumulative share 0.98.
6. Logit nodes: keep the top-$K$ tokens with total probability above 0.95, $K \le 10$.
7. Embedding and error nodes are never pruned.

Typical effect: nodes down by about 10x, edges down by about 500x; "we can generally reduce the number of nodes by an order of magnitude while reducing completeness by only 20%". Thresholds are "arbitrary and chosen manually".

Gotcha (inference): with $A[j,i]$ = weight from $i$ to $j$, each node's influence on the logits is $B^{T} w_{logit}$; the pseudocode writes `matrix_multiply(B, logit_weights)`, so check orientation when re-implementing.

### 9. Feature labelling and supernodes
- Features are labelled by hand from top-activating dataset examples, direct logit effects and graph neighbors; automated labels were used only as a starting point ("generally found human labels to be more reliable").
- Supernodes are manual groupings of (feature, position) nodes sharing the facet that matters on the prompt; members should "promote each other" and have same-sign downstream effects. Clustering by decoder vectors or adjacency was tried and found insufficient.
- Supernode edges: $\text{supernode\_adjacency}(t,s) = \dfrac{\sum_{n_t \in N_t} \text{frac\_external}(n_t) \cdot \sum_{n_s \in N_s} A'_{n_t,n_s}}{\sum_{n_t \in N_t} \text{frac\_external}(n_t)}$, where $A'$ is the input-normalized adjacency and $\text{frac\_external}(n_t)$ is the fraction of absolute input weight to $n_t$ coming from outside its supernode.

### 10. Intervention / steering validation
- Principle: nodes suggest which features matter and edges suggest how; validate by perturbing features in the underlying model and checking downstream features and outputs against the graph. A feature is intervened on "by modifying [its] computed activation and injecting its modified decoding in lieu of the original reconstruction".
- Constrained patching (default): choose a layer range; modify the feature's decoding at each layer in the range (MLP outputs are clamped to values precomputed once at the outset); run the forward pass from the last layer of the range. No second-order effects inside the range; later MLP outputs change only indirectly. Attention patterns are held at their unperturbed values.
- Multiplicative steering: multiply the feature activation by $M$. Standard inhibition: every feature in a supernode set to the negative of its original value ($M=-1$); report each supernode's total activation as a fraction of its original value, plus the logit distribution. The end layer of the range is chosen as the one giving the largest logit effect.
- Why not add decoder vectors at every layer during a normal forward pass: it double-counts, because the activity reconstructed at one layer may be causally upstream of that reconstructed at later layers.
- Iterative patching: recompute features that read from layers inside the range so knock-on effects are included; more complete but prone to cascading errors from repeated encode/decode, and less precise as a test of graph edges.
- What is actually tested: with frozen attention, direct feature-feature edges are "nearly forced to be confirmed" (guaranteed if the intervention end layer equals the target's encoder layer). The nontrivial test is of knock-on effects (A to B to C to token).
- Steering strengths are ad hoc: zero-ablation is often too weak, so features are clamped to negative multiples (example: one supernode needed -5); features added to a prompt are often set well above typical activation. Stated reasons: unexplained variance, incomplete supernodes, per-layer-range steering undervaluing a cross-layer feature's whole effect.
- Layer localization: sweep the end layer of the patching range with the start fixed at layer 1; the effect plateaus before the layers holding the downstream features, then shrinks once the range contains them.

### 11. Global weights
- Three path types from source $s$ to target $t$: residual-direct, attention-direct (through OV steps, scaled by attention patterns), indirect (through other features).
- Virtual weight (formula truncated in the extracted text; recovered from the local HTML): $V_{st} = \left\langle \sum_{\ell \in L_{st}} W_{dec}^{s,\ell},\; W_{enc}^{t} \right\rangle$, where $L_{st}$ is the set of layers with $\ell_s \le \ell < \ell_t$. An attribution edge = residual-direct term (virtual weight times activation) + attention-direct term.
- Problem: interference. Features that never co-activate can still have large virtual weights.
- Expected residual attribution: $V_{ij}^{\text{ERA}} = \mathbb{E}\left[\mathbb{1}(a_j > 0)\, V_{ij}\, a_i\right] = \mathbb{E}\left[\mathbb{1}(a_j > 0)\, a_i\right] V_{ij}$.
- Target-weighted expected residual attribution: $V_{ij}^{\text{TWERA}} = \dfrac{\mathbb{E}\left[a_j a_i\right]}{\mathbb{E}\left[a_j\right]}\, V_{ij}$.
- Both cover only the residual-direct component. TWERA "heavily relies on the coactivation statistics", gives strong values to some extremely small virtual weights, and "does not handle inhibition well".
- Alternative fix: restrict to features active on a narrow domain (addition: features active on at least 10 of 10,000 prompts).

### 12. Evaluation metrics
- CLT: normalized MSE (averaged across layers), L0 (average active features per token), replacement-model top-1 match with the underlying model, replacement-model KL divergence, and two LLM-judged interpretability scores. Sort Eval: Claude guesses which of two features produced held-out examples. Contrastive Eval: Claude guesses which of two near-identical prompts activated a feature.
- Baselines: per-layer transcoders on every layer; raw neurons with activations below a threshold clamped to 0.
- Path length: $B_{\ell} = \sum_{i=0}^{\ell} A^{i}$, and $P_{\ell} = \sum_{e} B^{\ell}_{t,e}$ over embedding nodes $e$ for logit node $t$ (probability-weighted over logit nodes), normalized by the total embedding influence in the unpruned graph.
- Graph completeness score: fraction of input edges (weighted by the target node's logit influence) coming from feature or embedding nodes rather than error nodes.
- Graph replacement score: fraction of end-to-end paths (weighted by strength) from embedding nodes to logit nodes that go via feature nodes rather than error nodes. For pruned graphs, pruned nodes count as error.
- Mechanistic faithfulness: (a) correlation between node logit influence and KL effect of ablating the node; (b) correlation between graph feature-to-feature influence and ablation effect on the downstream feature; (c) agreement of perturbation effects between the local replacement model and the underlying model (both with frozen attention), measured as cosine similarity and normalized MSE of the net perturbation (perturbed minus clean) at each downstream layer. Perturbation types: add a feature's encoder vector, add a random direction, or perturb an upstream feature and hand the residual state over at an intervention layer; perturbations scaled to raise the chosen feature's activation by 0.1.

## Experiments and numbers

### Models, dictionary sizes, data, compute
| Item | 18L | Claude 3.5 Haiku |
|---|---|---|
| Model | 18-layer transformer; no MLP in layer 0, so the CLT has 17 layers | frontier production model |
| Parameter count, $d_{model}$ | not stated | not stated |
| CLT total features | 300K to 10M (runs at 300K, 1M, 3M, 10M) | 300K to 30M |
| PLT baseline | same total feature counts | not stated |
| Training tokens (largest run) | about 3B | about 16B |
| Normalized MSE (largest run) | about 11.5% | 21.7% |
| L0 (largest run) | 88 | 235 |
| Top-1 match of replacement model | 50% (largest CLT) | not stated |
| Compute actually used | not stated | not stated |

Estimated CLT training cost for open-weights models (the authors' estimates, not reported runs):

| Model | Features | Train tokens | Training FLOPs | H100 hours at 50% FLOPs efficiency |
|---|---|---|---|---|
| Gemma 2 2B | 2M | 1B | 3.8e20 | roughly 210 |
| Gemma 2 9B | 5M | 3B | 6.9e21 | roughly 3,844 |

### CLT vs PLT vs neurons (18L)
- "On all metrics, CLTs outperform PLTs, and both CLTs and PLTs substantially outperform the Pareto frontier of thresholded neurons." Sort Eval and Contrastive Eval values: not stated (figures only).
- FLOPs: a PLT run costs about as much as the next smaller CLT (10M PLT is about 3M CLT). The 1M CLT "strictly outperforms" the 10M set of PLTs on (MSE, L0) and (replacement KL, L0) "using 5-10x less compute". At matched FLOPs (3M CLT vs 10M PLT): CLT has somewhat lower MSE and KL, roughly equal Sort Eval, worse Contrastive Eval (no Pareto winner declared).

| Prompt | Replacement model | Avg path length | Completeness | Replacement |
|---|---|---|---|---|
| capital-analogy | CLT (10M) | 2.3 | - | - |
| capital-analogy | PLT (10M) | 3.7 | - | - |
| uspto-telephone | CLT (10M) | - | 0.80 | 0.61 |
| uspto-telephone | PLT (10M) | - | 0.78 | 0.37 |

On "Zagreb:Croatia::Copenhagen:" the PLT graph has a length-7 path made entirely of Copenhagen features, which the CLT collapses into layer-1 features.

### Pruning threshold (one acronym prompt, "iasg")
| Node threshold | Completeness | Node count |
|---|---|---|
| 0.95 | 0.87 | 236 |
| 0.9 | 0.83 | 137 |
| 0.8 (default) | 0.70 | 55 |
| 0.7 | 0.58 | 27 |

### Graph sizes
- About a hundred active features per token position; unpruned graphs can have millions of edges even for short prompts.
- A pruned graph "often contains hundreds of nodes and tens of thousands of edges" (the limitations section says "hundreds of features and thousands of edges").

### Faithfulness
- Node-to-logit: 20 prompts, ablate every feature with constrained patching, compare KL to activation, direct logit edge and indirect influence by log-log Pearson correlation. Ordering: influence > direct edge > activation. Values: not stated (figure).
- Feature-to-feature: 20 prompts, constrained patching over layers $[i, i+2]$ ($i$ = encoder layer of the ablated feature): Spearman 0.72 between graph influence and normalized absolute ablation effect.
- Local replacement vs underlying model: one layer after the intervention about 0.8 cosine similarity and about 0.4 normalized MSE (main text); 10M dictionary cosine "around 60-80%" in the following layer (appendix). Discrepancies compound over layers; direction degrades gradually, magnitude can degrade "catastrophically", worse for larger dictionaries. PLTs: similar cosine, lower normalized MSE at one layer, slightly faster compounding in the largest PLTs.

### Evaluation data
- Feature visualizations: Common Corpus, The Pile (books removed), LMSYS Chat 1M, Isotonic Human-Assistant Conversation.
- Graph evaluation set: n=260 short sequences from the Pile (minus books), "due to the expense of graph generation". Filters: correct top-1 prediction; token density below 1e-4; target token not elsewhere in the context; loss above 0.2. The main text describes the top-1 match set as correctly predicted tokens with confidence below 80%.

### Case studies
- Acronym, 18L: "The National Digital Analytics Group (N" to "DAG". Three parallel word-to-"say _letter" paths; "National" has minimal influence (hypothesized to act through attention).
- Factual recall, 18L: "Fact: Michael Jordan plays the sport of" to "basketball" (65% confidence). Parallel "sport" and "Michael Jordan" paths, confirmed by supernode ablations.
- Addition, Haiku: "calc: 36+59=". Low-precision, moderate-precision and ones-digit pathways with lookup-table features. Suppressing the "_6" input supernode makes the model confidently output 98 instead of 95. Top-1 accuracy on the 10,000 two-digit prompts: 18L 98.6%, Haiku 100%. 2931 features in the 18L addition virtual-weight explorer.
- Interference analysis: 1000 sampled features, coactivation statistics over about 150M tokens; even among virtual weights beyond the 99.9999th percentile "a significant proportion are unused".

## What it establishes
- A complete, re-implementable recipe for per-prompt attribution graphs with exactly linear feature-to-feature edges given frozen attention and normalization.
- On 18L, by the paper's own metrics, CLTs give shorter paths and higher replacement scores than per-layer transcoders, and both beat thresholded neurons.
- Graph-derived indirect influence predicts ablation effects better than direct attribution or activation magnitude (20 prompts).
- In three case studies, supernode steering effects are "largely consistent" with the graphs.
- Local replacement models are reasonably faithful one layer downstream of a perturbation and much less so many layers downstream.
- Concrete failure cases exist: induction, a multiple-choice question (attention is the crux), an obfuscated jailbreak prompt (almost all error nodes).

## What it does NOT establish
- No results on any open-weights model: Gemma appears only in cost estimates. Nothing on Phi-3.5 or on medical QA.
- No comparison of a base model with its fine-tune in this paper (the fine-tuned "hidden goal" case is in the companion paper).
- Not a global circuit: the authors say the work "falls short by only offering a locally valid attribution graph".
- No explanation of how attention patterns are computed (QK circuits).
- Sufficiency of the hand-drawn supernode subgraphs is intentionally not measured; supernode coherence is not quantified.
- That the CLT uses the same mechanism as the MLPs: not guaranteed, only checked post hoc and coarsely.
- That PLT or neuron graphs are equally good: the paper only says they "still produce useful insights".
- Nothing about editing weights or transferring a computation between models.

## Limitations stated by the authors
- Missing attention circuits: graphs do not explain attention patterns and can "miss the interesting part"; in many cases this "renders our attribution graphs essentially useless". Multiple-choice example: the graph for the "B" answer token reduces to a "tokens following (b)" feature on the correct answer and "skips over" how the model knew 1945 was right. Known: "freezing attention locks the answer"; three "correct answer" features exist and activating them on option C makes the model answer "C". Unknown: which heads, and whether those features act on the key side.
- Reconstruction errors and dark matter: error nodes have no inputs and are uninterpretable; on an out-of-distribution prompt almost all nodes were error nodes. Proposed fixes include scaling, architecture changes, end-to-end training, "finetuning the replacement model on data distributions of interest", and attributing back from error nodes.
- Inactive features and inhibition: graphs show only active features; suppressed features can be the key. Workaround: contrastive prompt pairs, keeping features active in at least one prompt of each pair.
- Graph complexity: hundreds of features after pruning, concepts smeared across features, multiple paths of different signs.
- Features at the wrong level of abstraction: feature splitting and absorption; no single feature fired on all twelve prompts that Haiku completes with "during"; generality decreases with CLT size. Supernodes are a manual, lossy workaround.
- Global circuits are hard: interference weights; attention-mediated interactions are not in the basic global weights.
- Mechanistic faithfulness: validation is "very coarse" (supernodes, directional effects, best layer chosen by sweep, ad hoc steering strengths); predictive power is "reasonably good" a few layers downstream and "much worse" many layers downstream.
- CLTs have a different causal structure from the model (they hide layer-by-layer amplification chains), which can make them mechanistically unfaithful.
- Labels are error-prone: even for addition the authors mislabelled features from dataset examples alone.

## Implementation notes for med_wsp
- Released artifacts per the source: implementation guidance for CLTs, the pruning algorithm with pseudocode, and an MIT-licensed front end at https://github.com/anthropics/attribution-graphs-frontend. No model weights, CLT checkpoints or training code are mentioned.
- (background, not from the source, unverified here) An open-source `circuit-tracer` library implementing this pipeline with per-layer transcoders was released after this paper; check whether it supports Phi-3.5-mini and whether public transcoders exist before training our own.
- Target model facts (background, not from the source): Phi-3.5-mini has about 3.8B parameters, 32 layers, $d_{model}=3072$.

| Pipeline part | Fits on a 36 GB M4 Max? | Basis |
|---|---|---|
| Train a CLT | No | Source estimates roughly 210 H100-hours for a 2B model (2M features, 1B tokens) and roughly 3,844 for a 9B model; a 3.8B model sits between (inference). Memory (inference): each feature needs an encoder vector plus on average about 16.5 decoder vectors of size 3072, about 54K parameters per feature, so 1M features is about 54B parameters (about 215 GB in float32). |
| Train per-layer transcoders on a few chosen layers | Plausible, small dictionaries | (inference) 2 x 3072 parameters per feature; 16,384 features on one layer is about 0.1B parameters. All 32 layers at that size is about 3.2B parameters (about 13 GB float32). |
| Store activations on disk | No, beyond small samples | (inference) residual input plus MLP output in fp16 is about 12 KB per token per layer, about 123 GB per 10M tokens per layer. Generate activations on the fly with a shuffle buffer (the source says token-level shuffling matters). |
| Neuron-level graphs (no training) | Yes | Source: neurons need no extra compute and "even making attribution graphs with raw neurons will yield plenty of interesting insights"; they are less interpretable and thresholding damages behavior. |
| Attribution graph for one prompt | Yes, with adaptive / partial generation | Cost is one backward pass per target node, linear in active features. (inference) long MedQA prompts times about 100 features per position means tens of thousands of nodes, so use the queue-based variant and restrict target positions. |
| Pruning, influence matrix | Yes for thousands of nodes | One dense inverse $(I-A)^{-1}$ (inference on feasibility). |
| Constrained patching and steering | Yes | Forward passes only. |
| LLM-judged interpretability evals, n=260 graph evaluation | Reduce scale | Source used n=260 because graph generation is expensive. |

- Gotchas (inference unless quoted): we need the attention pattern itself and the normalization denominators as detachable tensors, so fused attention kernels must be replaced by an explicit implementation; MLP outputs need stop-gradients; error nodes must be recomputed per prompt; with PLTs the decoder sum in the edge formula has a single term ($\ell = \ell_s$).
- For a logit-difference question (correct letter minus wrong letter), the output-node input vector can be the gradient of that difference instead of logit minus mean logit (inference; the source only defines the mean-centered logit).
- The source's multiple-choice failure case matches our prompt format. Expect letter-logit graphs on MedQA to be dominated by attention-mediated edges from option tokens (inference). Head-level [[Activation patching]] is the complement; "correct answer"-style features at option positions are candidate attribution targets (inference).

## Relevance to med_wsp (provisional)
- Provisional idea: use [[Attribution graphs]] built on per-layer [[Transcoders]] (or neurons) for layers that [[Activation patching]] and weight diffs flag as carrying the MediPhi gain, instead of a full CLT.
- Provisional idea: apply one dictionary to both Phi-3.5 and MediPhi ([[Corbeil et al. 2025 - MediPhi]]) and compare graphs; the fine-tune-specific computation would show up as changed feature activations and larger error nodes (inference, untested). The source mentions an internal "diff view comparing two attribution graphs" that was trimmed from the public interface. Whether a base-model transcoder transfers to the fine-tune is the question [[Tan et al. 2026 - Efficient Transcoder Adaptation]] may address (check that note; not verified here), and [[Lindsey et al. 2024 - Sparse Crosscoders]] / [[Crosscoders]] is the alternative for model diffing.
- Provisional idea: the contrastive-pairs strategy for inactive features maps onto our setting as "same prompt, two models" (inference).
- Provisional idea: constrained patching, steering sweeps, and the completeness / replacement scores are the validation template for any circuit GPT-6 Astra proposes before weight edits ([[Weight-space programming]], [[Takehi et al. 2026 - Weight-Space Programming]], [[Li et al. 2026 - WeightCode]]).
- Caution: the paper gives no evidence that a circuit found this way can be transplanted into another model.

## Open questions
- How much of the Phi-3.5 to MediPhi difference on MedQA is attention-mediated (QK), which this method cannot see?
- What replacement and completeness scores do small per-layer transcoders reach on Phi-3.5 MedQA prompts, and is that enough to trust a graph? The source gives no acceptance threshold.
- Does a transcoder trained on Phi-3.5 keep low error on MediPhi activations?
- Is fine-tuning transcoders on medical text (a future direction in the source) needed to avoid error-node-dominated graphs on clinical vignettes?
- How should supernodes be formed without manual labelling at our scale? The source found no adequate automated method.
- Which steering strength counts as a fair test, given the source's ad hoc choices?

## Ledger card
**Full citation:** Ameisen, E., Lindsey, J., Pearce, A., Gurnee, W., Turner, N. L., Chen, B., Citro, C., et al. (2025). "Circuit Tracing: Revealing Computational Graphs in Language Models". Transformer Circuits Thread, Anthropic. Published March 27, 2025.
**Source / DOI / URL:** https://transformer-circuits.pub/2025/attribution-graphs/methods.html (no DOI stated); local text `papers/pdf/text/circuit_tracing_methods_2025.txt`.
**Why I read it:** It defines the attribution-graph method we plan to adapt for finding what MediPhi computes differently from Phi-3.5.
**Problem addressed:** How to describe, per prompt, the computational steps of a language model in terms of interpretable sparse features with well-defined linear interactions, and how to validate such descriptions.
**Method / representation:** Cross-layer transcoder replacing MLPs; local replacement model with frozen attention patterns and normalization denominators plus error nodes; edges $A_{s\to t} = a_s w_{s\to t}$ from stop-gradient backward Jacobians; influence-based pruning; manual supernodes; constrained-patching interventions; virtual weights and TWERA.
**Assumptions:** Features are meaningful units; attention patterns can be treated as fixed; active features carry the mechanism; the transcoder approximates MLP computation by a similar mechanism (checked only post hoc).
**Evidence used by the paper:** experiments: CLT / PLT / neuron comparisons on 18L, case studies (acronym, factual recall, addition), steering interventions, faithfulness perturbation studies / analysis: graph metrics (influence, path length, completeness, replacement), global weights / datasets or systems: 18L research model, Claude 3.5 Haiku, Pile (minus books) evaluation set of n=260, Common Corpus, LMSYS Chat 1M, Isotonic Human-Assistant Conversation.
**Safe citation claims:** "Ameisen et al. (2025) construct per-prompt attribution graphs on a local replacement model in which MLPs are replaced by a cross-layer transcoder and attention patterns and normalization denominators are frozen"; "they report that their largest 18L CLT matches the model's top-1 next token on 50% of a filtered set of prompts"; "they state that attribution graphs do not explain attention patterns and can miss the crux of multiple-choice behavior"; "they state that per-layer transcoders or MLP neurons can be substituted for the CLT while keeping the rest of the pipeline".
**Unsafe / unsupported citation uses:** "attribution graphs give a complete or faithful account of model computation"; "the method was validated on open-weights models or on medical QA"; "CLTs are required for circuit tracing"; "the paper shows circuits can be transferred between models or edited in weights"; any specific Sort Eval / Contrastive Eval value (figures only).
**Relation to our project:** overlapping method (the attribution-graph step of our pipeline); also supporting source for validation methodology.
**Verification status:** full text read: yes (extracted text; three truncated formulas and one pseudocode line recovered from the local HTML; figures not viewed); key claim checked: yes, against the extracted text only.

## Links
- Concepts: [[Attribution graphs]], [[Activation patching]], [[Transcoders]], [[Crosscoders]], [[Weight-space programming]]
- Papers: [[Lindsey et al. 2024 - Sparse Crosscoders]], [[Tan et al. 2026 - Efficient Transcoder Adaptation]], [[Corbeil et al. 2025 - MediPhi]], [[Takehi et al. 2026 - Weight-Space Programming]], [[Li et al. 2026 - WeightCode]]
