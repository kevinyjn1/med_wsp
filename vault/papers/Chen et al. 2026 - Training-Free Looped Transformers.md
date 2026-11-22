---
type: paper
title: "Training-Free Looped Transformers"
short: "Chen et al., 2026"
authors: [Lizhang Chen, Jonathan Li, Chen Liang, Ni Lao, Qiang Liu]
year: 2026
venue: "arXiv:2605.23872v1 [cs.LG], preprint, 22 May 2026"
url: "https://arxiv.org/abs/2605.23872"
local_text: "papers/pdf/text/arxiv_2605.23872.txt"
tags: [paper, layer-looping, training-free, inference-time, ode-view, kv-cache]
read_status: full-text-read
---

# Chen et al., 2026 - Training-Free Looped Transformers

## TL;DR
An inference-time wrapper re-applies a contiguous mid-stack window of layers $[a,b]$ of a frozen checkpoint $K$ times, with no weight updates, no new parameters and no architecture change. Naive reapplication $x \leftarrow g(x)$ degrades accuracy, so the paper reads a pre-norm block as one forward-Euler step ($h=1$) on $\dot x = F_g(x)$ with $F_g(x)=g(x)-x$, and replaces that one step by $K$ damped sub-steps $x_{k+1} = (1-\tfrac1K)x_k + \tfrac1K g(x_k)$, optionally blended with the ordinary one-step output through an anchor weight $\beta$. The recipe is a 4-layer window whose center sits at depth fraction about 0.45-0.60, $K$ = 2 or 3, block-mode for dense models and layer-mode for MoE models. Headline gains (best configuration per cell): +2.64 pp MMLU-Pro and +2.01 pp GPQA-Main on Qwen3-4B-Instruct, +2.30 pp ARC-Challenge on Qwen1.5-MoE-A2.7B; 27 of 45 (model, benchmark) cells are positive (> +0.3 pp) and 12 neutral. One medical benchmark appears: MedMCQA 0-shot on Qwen3-4B-Instruct, 53.40 -> 54.13 (+0.73 pp).

## Problem
Looped transformers, Universal Transformers and Deep Equilibrium models build recurrence into training, so looping is normally unavailable for checkpoints that were not trained that way, which is nearly every released LLM. The paper asks whether a frozen, off-the-shelf checkpoint can be looped directly at inference time with no fine-tuning, no continued training, no auxiliary parameters and no architectural change. The motivation is prior evidence that mid-layer blocks can be deleted, skipped, swapped or repeated without catastrophic loss; the authors argue that the redundancy that makes a mid layer safe to remove also makes it safe to apply again.

## Method (step by step)

### 1. Loop wrapper
- Network $f = L_{N-1} \circ \dots \circ L_0$ with $N$ decoder layers, each $L_i : \mathbb{R}^{T \times d} \to \mathbb{R}^{T \times d}$ (the whole residual stream, all $T$ positions).
- Choose a contiguous window $[a,b]$ (0-indexed, inclusive, $0 \le a \le b \le N-1$) and loop count $K \ge 1$. Window operator (eq. 1): $g := L_b \circ L_{b-1} \circ \dots \circ L_a$. Window width $W = b-a+1$.
- Looped network (eq. 2): $\hat f(x) = (L_{N-1} \circ \dots \circ L_{b+1}) \circ g^{(K)} \circ (L_{a-1} \circ \dots \circ L_0)(x)$. Pre-loop and post-loop layers run once, unchanged.
- Stated cost (Figure 1 caption): $(b-a+1)(K-1)$ extra layer forward passes through the window.

### 2. Iteration mode
- Block-mode (eq. 3): $g^{(K)}_{blk}(x) = (L_b \circ \dots \circ L_a)^K(x)$ - the whole window is iterated as one unit.
- Layer-mode (eq. 4): $g^{(K)}_{lyr}(x) = L_b^K \circ L_{b-1}^K \circ \dots \circ L_a^K(x)$ - each layer is iterated $K$ times before passing on.
- Dense models: the paper says the two modes give broadly similar quality; block-mode is the default. MoE models: in block-mode the gating network sees a slightly perturbed state at each iteration and routes to different experts ("routing thrash"); layer-mode is the stated default for MoE backbones.

### 3. ODE view
- Pre-norm layer (eq. 5): $L(x) = x + \mathrm{Attn}(\mathrm{LN}_1(x)) + \mathrm{MLP}(\mathrm{LN}_2(x + \mathrm{Attn}(\mathrm{LN}_1(x))))$.
- Window residual field (eq. 6): $F_g(x) := g(x) - x$, so $g(x) = x + F_g(x)$ is one forward-Euler step with $h=1$ (eq. 7: $x_1 = x_0 + h \cdot F_g(x_0)$) on the autonomous ODE $\dot x = F_g(x)$ (eq. 8).
- In block-mode the field telescopes (eq. 9, proof in App. B.1): $F_g(x) = \sum_{i=a}^{b} F_{L_i}(y_i(x))$, with $y_a(x) := x$, $y_{i+1}(x) := y_i(x) + F_{L_i}(y_i(x))$, $F_{L_i}(z) := L_i(z) - z$.
- Key argument: the post-loop layers were trained to receive the approximation of $x(t{=}1)$ produced by one application of $g$. Naive looping is $K$ Euler steps of size $h=1$ and lands at $x(t{=}K)$, which the post-loop layers never saw.

### 4. Update rules
**Naive loop** (Table 1): $x_{k+1} = g(x_k)$, applied $K$ times. $K$ forward passes through the window.

**Damped sub-steps / forward Euler with $h = 1/K$** (eq. 10, the core rule):

$$x_{k+1} = \left(1 - \frac{1}{K}\right) x_k + \frac{1}{K}\, g(x_k) \;=\; x_k + \frac{1}{K} F_g(x_k), \qquad k = 0,\dots,K-1$$

- Splitting: the one block update is split into $K$ sub-steps over the same total integration time $[0,1]$.
- Damping coefficient: $\alpha = h = 1/K$ (the notation table gives $\alpha \in (0,1]$ and identifies it with the step size $h$). General form used in Algorithm 4: $x \leftarrow x + \alpha (y - x)$ with $y = g(x)$. Ablations also use $\alpha \ne 1/K$ (for example $K{=}2$, $\alpha{=}0.7$) and per-step schedules (for example $\alpha = [0.6, 0.4]$).
- How residual and block output are combined: a convex combination of the current iterate (weight $1-\tfrac1K$) and the full window output $g(x_k)$ (weight $\tfrac1K$).
- Attention recomputation: every sub-step evaluates the full $g$ (attention and MLP of every window layer) on the current iterate; Table 1 lists $K$ forward passes for forward Euler. (inference from eqs. 6, 10, 14: nothing in the text reuses attention outputs across sub-steps. In block-mode each layer inside one $g$ evaluation still takes its ordinary full step; only the composite window residual is scaled by $1/K$. In layer-mode the $1/K$ scaling and the $K$ evaluations are per layer.)
- Claimed property: converges to the true $x(t{=}1)$ at order $O(1/K)$.

**$K$-stage Runge-Kutta with anchor $\beta$** (eqs. 11-16, Algorithm 3; this is the method labeled "ours"):
- General explicit RK (eqs. 11-12): $x_1 = x_0 + h \sum_{i=1}^{s} b_i k_i$, $k_1 = F_g(x_0)$, $k_i = F_g\!\left(x_0 + h \sum_{j<i} a_{ij} k_j\right)$.
- Chosen tableau (eq. 13): $h = 1$, $s = K$, $a_{ij} = \tfrac{1}{K}\mathbb{I}(j<i)$.
- Stages (eq. 14): $y_i = x_0 + \tfrac1K \sum_{j=1}^{i-1} k_j = F^{\,i-1}(x_0)$, $k_i = F_g(y_i)$, where $F(x) := x + \tfrac1K F_g(x)$ is the damped Euler step.
- Output weights (eq. 15): $b_1 = \beta + \tfrac{1-\beta}{K}$, $b_i = \tfrac{1-\beta}{K}$ for $i = 2,\dots,K$, with $\beta \in [0,1]$.
- Resulting identity (eq. 16, proof in App. B.2):

$$x_1 = x_0 + \sum_{i=1}^{K} b_i k_i = \beta\, g(x_0) + (1-\beta)\, F^{K}(x_0)$$

- $\beta = 0$ recovers the $K$-sub-step forward Euler endpoint; $\beta = 1$ returns the original output $g(x_0)$.
- Algorithm 3 as printed (checked against the PDF; block-mode / layer-mode variants overlaid): `x <- PreLoop(x)`; anchor `x~ <- (L_b ... L_a)(x)` (block) or, inside `for l = a..b`, `x~ <- L_l(x)` (layer); `for k = 1..K-1: y <- g(x) (or L_l(x)); x <- x + (1/K)(y - x)`; `x <- beta * x~ + (1 - beta) * x`; `return PostLoop(x)`.
- Source discrepancy: the printed inner loop runs $k = 1,\dots,K-1$, that is $K-1$ damped steps after a separate anchor evaluation, while eq. 16 needs $F^K$ ($K$ damped steps). Reusing the anchor evaluation $g(x_0)$ as the first sub-step gives eq. 16 with exactly $K$ window evaluations (inference).
- The value of $\beta$ used for the headline numbers is not stated. App. E refers once to the K=2-stage Runge-Kutta of Algorithm 3 with β=0.5, and the per-cell table labels the same kind of runs "Euler".

**Other strategies evaluated** (Table 1; forward passes in brackets):
- Midpoint RK2: $x_{k+1} = x_k + h F_g(x_k + \tfrac h2 F_g(x_k))$ [2K]
- Heun RK2: $x_{k+1} = x_k + \tfrac h2 (F_g(x_k) + F_g(x_k + h F_g(x_k)))$ [2K]
- RK4: $x_{k+1} = x_k + \tfrac h6(\kappa_1 + 2\kappa_2 + 2\kappa_3 + \kappa_4)$ [4K]
- Heavy-ball $(\alpha,\beta)$: $x_{k+1} = x_k + \alpha F_g(x_k) + \beta (x_k - x_{k-1})$ [K]
- Anderson $(m,\beta)$: $x_{k+1} = (1-\beta)(x_k - (\Delta X_k)\gamma_k^\star) + \beta(g(x_k) - (\Delta F_k)\gamma_k^\star)$ [K]
- Aitken $\Delta^2$: $x_{k+1} = x_k - \tfrac{(d_{1,k})^2}{d_{2,k}}$ per coordinate, $d_{1,k} = g(x_k) - x_k$, $d_{2,k} = g(g(x_k)) - 2g(x_k) + x_k$, Steffensen safeguard, $K$ even [K]
- Uniform loop (this is [[Lys et al. 2026 - Inner Loop Inference]] as rendered by Chen et al.): $x_{k+1} = g\!\left(\tfrac{1}{k+1}\sum_{i=0}^{k} x_i\right)$ [K]
- Also in ablations, described only by name: `norm_stab` (rescales each iterate to a fixed L2 norm), `poly_blend` with weight vectors $w$ (definition not stated), Euler schedules, a "halt τ=0.05" option (not defined), and "EMA" in the search space (not defined).

### 5. Choosing the window (start, length)
- Length: $n = 4$ layers. Evidence: width sweep on Qwen3-1.7B-Base (Table 2a) and $n{=}3$ vs $n{=}4$ on four models (Table 2b). Accuracy falls off sharply at $n \ge 6$.
- Position: the "depth fraction rule". With window center $(a+b)/(2N)$: for models larger than 1.7B the optimum is reported at 0.43-0.71 (mode about 0.50); for sub-1B models it shifts earlier, 0.25-0.56. Figure 6: the 0.45-0.60 band contains 7 of 9 checkpoints' optimal centers; Llama-3.2-1B is earlier and Qwen1.5-MoE-A2.7B later. The conclusion states the rule as about 0.45-0.60 for dense and MoE architectures with 16 to 48 layers.
- Per-cell search protocol (App. P), which conflicts with the "no per-cell tuning" wording elsewhere (see inconsistencies): per cell, the Cartesian product of window center in $[0.40N, 0.70N]$ x width $\{3, 4, 6\}$ x $K \in \{2, 3\}$ x strategy $\{$NAIVE, EMA, EULER$\}$ x cache $\{$FIRST, LAST$\}$, plus a layer-mode counterpart where applicable.
- Validation: two-stage. Stage 1 screens each candidate on a "16-task aggregate" (composition not stated). Stage 2 re-scores the top-3 candidates per cell on MMLU 5-shot (about 14k examples), which the authors treat as held-out. Only configurations that are non-negative on both are kept. Stopping rule: best held-out score not improved over the last six configurations, or more than 8 GPU-hours on H100-80 per cell. About 720 (cell, configuration) pairs were visited.
- So a validation step is used, but it is a different benchmark (MMLU 5-shot), not a held-out split of the target benchmark, and the Table 6 caption describes each row as the best configuration found for that cell (inference: the per-cell numbers are selected on the reported benchmark itself). The only configuration described as committed before scoring is Qwen3-30B-A3B-Instruct at $[22,24]$ (Table 4 / Table 17).

### 6. KV cache and generation (App. C, N)
- Prefill (`seq_len > 1`, no past KV): the loop body runs with `use_cache=False` and writes nothing; one extra "stash" pass through the window after the loop writes the KV entries.
- Decode (`seq_len == 1`, past KV exists), Algorithm 4 (block-mode): run pre-loop layers normally; save $x_a$; snapshot the cache length $\ell_i = |C_i|$ of every loop layer; for $k = 1..K$: run layers $a..b$ with `use_cache=True` so attention reads the real past KV, then crop each $C_i$ back to $\ell_i$, then apply the strategy update $x \leftarrow S(x, y, k)$ (for example $x + \alpha(y - x)$); stash phase: one more pass through layers $a..b$ writes exactly one KV entry per loop layer; post-loop layers run normally from $x$ (the stash-pass output is not used downstream).
- Cache strategy $c$ (eq. 19) = input to the stash pass: LAST = post-loop state $g^{(K)}(x_a)$; FIRST = pre-loop input $x_a$; NONE = no write (an ablation that fails badly).
- Reason for snapshot/crop: the Hugging Face `DynamicCache.update()` always appends, so a plain $K$-iteration body would write $K$ extra entries per loop layer at one position and corrupt later attention.
- Total cache delta is exactly $b-a+1$ entries per token regardless of $K$, identical in shape to the unmodified model.
- Layer-mode (Algorithm 5): same idea with a per-layer snapshot/crop and a per-layer stash write.
- `decode_mode`: `bypass` loops only during prefill (default for log-likelihood-only evaluations); `full` loops at every decode step (default for the generation results in the main paper); `first_n` loops only the first $n$ generated tokens (reported as never better than `full` when looping helps).
- Guidance: `cache=first` is better for long-prompt chain-of-thought, `cache=last` for short structured generation.

### 7. Implementation
- The looped forward is a reversible monkey patch on the model's top-level decoder class (for example `Qwen3Model.forward`, `LlamaModel.forward`); stateless apart from KV writes and a $\Theta(W)$ activation buffer. Reloading the model with `from_pretrained()` restores exact baseline outputs.
- Software: lm-evaluation-harness v0.4.11, transformers 4.46, bfloat16 weights, SDPA attention, deterministic seeds.
- Code repository: not stated.

## Experiments and numbers

### Setup
- Models (layer counts from Figure 6): Qwen3-0.6B (28), Qwen3-1.7B (28), Qwen3-4B (36), base and instruct; Qwen3-30B-A3B-Instruct MoE (48); Qwen1.5-MoE-A2.7B-Chat (24); Llama-3.2-1B-Instruct (16) and Llama-3.2-3B-Instruct (28); DeepSeek-V2-Lite-Chat (27, MLA + 64-expert MoE); Moonlight-16B-A3B-Instruct (27, MLA + MoE).
- Benchmarks named: MMLU-Pro, GPQA-Main, GPQA-Diamond, CommonsenseQA (CSQA), MMLU (0-shot, 5-shot, "em", "flex", "1500"), MMLU-Redux, ARC-Challenge, ARC-Easy, OpenBookQA (OBQA), SciQ, MedMCQA, C-Eval, HellaSwag, TruthfulQA-MC1, SuperGPQA (n=2000), LAMBADA (accuracy and perplexity), GSM8K, MBPP, plus a "16-task" suite and a "19-task held-out suite" whose compositions are not stated.
- Baselines: the unmodified checkpoint (no loop), naive loop, uniform loop of [[Lys et al. 2026 - Inner Loop Inference]], higher-order RK solvers, heavy-ball, Anderson, Aitken, norm_stab, poly_blend.
- Stated recipe (Section 3.1): 3-stage Runge-Kutta on the mid 4 layers, block-mode for dense and layer-mode for MoE. App. D instead names K=2-stage Runge-Kutta with n=4 and cache=first as the robust default.
- Total compute: approximately 20,000 NVIDIA H100 (80 GB) GPU hours.
- Statistics: point estimates only. Reported run-to-run variance under fixed seeds: ≤0.02 pp on benchmarks with ≥10,000 examples (MMLU, MMLU-Pro, ARC-Challenge); ≤0.5 pp on benchmarks with ≤1,000 examples (GPQA-Main, OpenBookQA). abs(Δ) ≤ 0.3 pp is treated as neutral.

### Headline table (Table 3; accuracies as fractions, quoted exactly)
| Backbone type | Model | Benchmark | Base | Loop (Ours) | Δpp |
|---|---|---|---|---|---|
| Dense | Qwen3-4B-Instruct | MMLU-Pro 5-shot | 0.5714 | 0.5979 | +2.64 |
| Dense | Qwen3-4B-Instruct | GPQA-Main 0-shot | 0.3371 | 0.3571 | +2.01 |
| Dense | Qwen3-4B-Instruct | CommonsenseQA 7-shot | 0.7887 | 0.7993 | +1.06 |
| Dense | Llama-3.2-3B-Instruct | GPQA-Main 0-shot | 0.2991 | 0.3103 | +1.12 |
| Dense | Llama-3.2-3B-Instruct | MMLU-Pro 5-shot | 0.3164 | 0.3236 | +0.71 |
| Dense | Llama-3.2-3B-Instruct | MMLU 0-shot | 0.5966 | 0.6039 | +0.72 |
| Dense | Llama-3.2-1B-Instruct | GPQA-Main 0-shot | 0.2790 | 0.2969 | +1.79 |
| MoE | Qwen1.5-MoE-A2.7B | ARC-Challenge 25-shot | 0.4829 | 0.5060 | +2.30 |
| MoE | Qwen1.5-MoE-A2.7B | CommonsenseQA 7-shot | 0.7961 | 0.8133 | +1.72 |
| MoE | Qwen1.5-MoE-A2.7B | OpenBookQA 0-shot | 0.3160 | 0.3320 | +1.60 |
| MoE | Moonlight-16B-A3B | MMLU 0-shot | 0.6786 | 0.6860 | +0.74 |
| MoE | Moonlight-16B-A3B | OpenBookQA 0-shot | 0.3160 | 0.3280 | +1.20 |
| MoE | DeepSeek-V2-Lite-Chat | ARC-Challenge 25-shot | 0.5794 | 0.5879 | +0.85 |

Cross-architecture summary (Section 3.5): of 45 cells, 27 are positive (Δ > +0.3 pp, 60%) and 12 neutral (abs(Δ) ≤ 0.3 pp, 27%), so 87% are positive or neutral. (inference: that leaves 6 negative cells, 13%.) MLA-based MoE models move in the same direction with 3-4x smaller magnitudes.

### Per-cell best configurations (Table 6; "-" = not given in the source)
Format: window [start-end], K, strategy; cache / mode / decode.

| Model | Benchmark | Baseline | Loop | Δpp | Window / K / strategy | cache / mode / decode |
|---|---|---|---|---|---|---|
| Qwen3-4B-Instruct | MMLU-Pro 5sh CoT | 57.14 | 59.79 | +2.64 | [15-18] K=3 Euler | first / block / full |
| Qwen3-4B-Instruct | GPQA-Main 0sh | 33.71 | 35.71 | +2.01 | [15-18] K=3 Euler | first / block / full |
| Qwen3-4B-Instruct | SciQ 0sh | 93.30 | 95.10 | +1.80 | [15-18] K=3 Euler | first / block / full |
| Qwen3-4B-Instruct | MMLU 0sh | 68.15 | 69.45 | +1.30 | [15-18] K=3 Euler | first / block / full |
| Qwen3-4B-Instruct | CommonsenseQA 7sh | 78.87 | 79.93 | +1.06 | [15-18] K=3 Euler | first / block / full |
| Qwen3-4B-Instruct | MedMCQA 0sh | 53.40 | 54.13 | +0.73 | [11-14] K=3 Euler | first / block / full |
| Qwen3-4B-Instruct | ARC-Challenge 25sh | 62.54 | 63.14 | +0.60 | [15-18] K=3 damped Euler α=0.5 halt τ=0.05 | first / block / full |
| Qwen3-4B-Instruct | C-Eval 5sh | 72.21 | 72.73 | +0.52 | [15-18] K=3 Euler | first / block / full |
| Qwen3-4B-Instruct | OpenBookQA 0sh | 40.20 | 40.60 | +0.40 | [11-14] K=3 Euler | first / block / full |
| Qwen3-4B-Base | MMLU 5sh | 73.27 | 73.61 | +0.34 | [15-18] K=3 Euler | first / block / - |
| Qwen3-4B-Base | 16-task macro | 63.98 | 65.03 | +1.05 | [15-18] K=3 Euler | first / block / - |
| Qwen3-1.7B-Base | MMLU 5sh | 62.68 | 63.01 | +0.34 | [12-15] K=2 damped Euler | last / block / - |
| Qwen3-1.7B-Base | 16-task macro | 55.60 | 56.11 | +0.51 | [12-15] K=2 damped Euler | last / block / - |
| Qwen3-0.6B-Base | 16-task macro | 47.87 | 48.43 | +0.56 | [12-15] K=2 heavy-ball α=0.5, β=0.5 | last / layer / - |
| Llama-3.2-3B-Inst | MMLU-Pro 5sh CoT | - | - | +0.71 | [12-15] K=2 Euler | first / block / full |
| Llama-3.2-3B-Inst | GPQA-Main 0sh | - | - | +0.67 | [12-15] K=2 Euler | first / block / full |
| Llama-3.2-3B-Inst | MMLU 5sh | - | - | +0.72 | [16-19] K=2 Euler | last / layer / bypass |
| Llama-3.2-3B-Inst | OpenBookQA 0sh | - | - | +0.40 | [20-23] K=3 Euler | last / block / bypass |
| Qwen1.5-MoE-A2.7B | ARC-Challenge 25sh | 48.29 | 50.59 | +2.30 | [13-16] K=3 Euler | first / block / full |
| Qwen1.5-MoE-A2.7B | CommonsenseQA 7sh | 79.61 | 81.33 | +1.72 | [10-13] K=2 Euler | first / block / full |
| Qwen1.5-MoE-A2.7B | OpenBookQA 0sh | 31.60 | 33.20 | +1.60 | [14-17] K=3 Euler | first / block / full |
| Qwen1.5-MoE-A2.7B | SciQ 0sh | 94.80 | 95.10 | +0.30 | [14-17] K=3 Euler | first / block / full |
| DeepSeek-V2-Lite | ARC-Challenge 25sh | 57.94 | 58.79 | +0.85 | [13-16] K=2 Euler | first / layer / bypass |
| DeepSeek-V2-Lite | OpenBookQA 0sh | 35.80 | 36.60 | +0.80 | [13-16] K=2 Euler | first / block / bypass |
| DeepSeek-V2-Lite | MMLU (1500) | - | - | +0.56 | [10-13] K=3 Euler | first / block / bypass |
| DeepSeek-V2-Lite | CommonsenseQA 7sh | 75.43 | 75.92 | +0.49 | [11-14] K=2 Euler | first / block / bypass |
| Moonlight-16B-A3B | OpenBookQA 0sh | - | - | +1.20 | [8-11] K=3 Euler | first / block / bypass |
| Moonlight-16B-A3B | GPQA-Main 0sh | - | - | +0.89 | [15-18] K=2 Euler | first / layer / bypass |
| Moonlight-16B-A3B | ARC-Challenge 25sh | - | - | +0.51 | [11-14] K=2 Euler | first / layer / bypass |
| Moonlight-16B-A3B | CommonsenseQA 7sh | - | - | +0.49 | [11-14] K=2 Euler | first / layer / bypass |
| Qwen3-30B-A3B-Inst | CommonsenseQA 7sh | 78.71 | 79.85 | +1.14 | [22-24] K=2 Euler | first / block / bypass |

Llama-3.2-1B-Instruct is not in Table 6; the text gives GPQA-Main +1.79 at window [4-7] (other settings not stated). "Euler" with a given $K$ corresponds to the forward-Euler row of Table 1 (step $1/K$); Table 7 labels the Qwen3-4B-Base $K{=}3$ run as K=3, α≈1/3.

### Window selection (Table 2)
(a) Width sweep, Qwen3-1.7B-Base, forward Euler K=2:

| n | Window | 16-task | Δ |
|---|---|---|---|
| 1 | [14] | 55.72 | +0.18 |
| 2 | [13,14] | 55.60 | +0.06 |
| 3 | [13,14,15] | 55.75 | +0.21 |
| 4 | [12-15] | 56.09 | +0.55 |
| 6 | [11-16] | 54.72 | -0.82 |
| 12 | [8-19] | 54.91 | -0.63 |
| 28 (all) | [0-27] | 27.81 | -27.73 |

(b) n=3 vs n=4 across models:

| Model | n=3 Δ | n=4 Δ | Winner |
|---|---|---|---|
| Qwen3-0.6B-Base | +0.57 | +0.22 | n=3 |
| Qwen3-1.7B-Base | +0.21 | +0.55 | n=4 |
| Qwen3-4B-Base | +0.43 | +0.85 | n=4 |
| Llama-3.2-3B-Instruct | +0.23 | +0.45 | n=4 |

(c) Block-mode vs layer-mode on MoE at window [13, 16], Δpp vs baseline (the column header in the PDF says layer K=3, the caption says layer K=2):

| Model | Benchmark | block (K=3) | layer |
|---|---|---|---|
| Qwen1.5-MoE-A2.7B | ARC-Challenge | +0.17 | +0.85 |
| Qwen1.5-MoE-A2.7B | CSQA | +0.16 | +0.33 |
| Qwen1.5-MoE-A2.7B | OBQA | +1.00 | -1.40 |
| Qwen1.5-MoE-A2.7B | MMLU | -1.18 | 0.00 |
| Qwen1.5-MoE-A2.7B | SciQ | +0.20 | not run |
| Qwen1.5-MoE-A2.7B | GPQA-Main | -1.11 | not run |
| Moonlight-16B-A3B | ARC-Challenge | -1.45 | +0.51 |
| Moonlight-16B-A3B | CSQA | -0.66 | +0.49 |
| Moonlight-16B-A3B | OBQA | -0.20 | +1.20 |
| Moonlight-16B-A3B | MMLU | +0.74 | not run |
| Moonlight-16B-A3B | SciQ | -0.50 | -0.60 |
| Moonlight-16B-A3B | GPQA-Main | -2.00 | +0.90 |

Breadth of the position optimum (Table 19; Euler K=2, block-mode, n=4 windows, 16-task macro):

| Model | Layers | # positive n=4 wins | Best window | Worst still-positive |
|---|---|---|---|---|
| Qwen3-0.6B-Base | 28 | ≥ 6 of 13 tested | [8-11] +1.33 | [11-14] +0.17 |
| Qwen3-1.7B-Base | 28 | 9 of 17 tested | [7-10] +0.80 | [18-21] +0.02 |
| Qwen3-4B-Base | 36 | 13 of 16 tested | [15-18] +0.84 | [18-21] +0.07 |

On Qwen3-4B-Base the 13 positive windows span depth fractions 0.19-0.83. Qwen3-0.6B-Base windows beating baseline on the 16-task macro (App. J.3): [8-11] +1.33, [14-17] +0.86, [6-9] +0.85, [7-10] +0.66, [3-6] +0.46, [9-12] +0.36.

### Leakage-free transfer: Qwen3-30B-A3B-Instruct at [22-24] (Tables 4 and 17)
Configuration fixed in advance from the depth-fraction rule; K=2 (called "K-stage RK" in Table 4 and "Euler K=2" in Table 17), block-mode; alternative Heun K=1.

| Benchmark | Baseline | Euler K=2 | Δ | Heun K=1 | Δ |
|---|---|---|---|---|---|
| ARC-Easy | 79.04 | 79.38 | +0.34 | 79.08 | +0.04 |
| HellaSwag | 77.74 | 77.93 | +0.19 | 77.78 | +0.04 |
| SciQ | 94.80 | 95.00 | +0.20 | 94.90 | +0.10 |
| CommonsenseQA | 78.71 | 79.85 | +1.14 | 79.61 | +0.90 |
| TruthfulQA-MC1 | 34.15 | 34.64 | +0.49 | 34.27 | +0.12 |
| MMLU (em) | 81.75 | 81.75 | 0.00 | 82.28 | +0.53 |
| MMLU (flex) | 66.67 | 67.46 | +0.79 | 66.67 | 0.00 |
| GPQA-Main | 37.28 | 37.50 | +0.22 | 37.72 | +0.44 |
| GPQA-Diamond | 36.36 | 34.85 | -1.51 | 36.87 | +0.51 |
| SuperGPQA (n=2000) | 31.00 | 31.70 | +0.70 | 31.05 | +0.05 |
| LAMBADA accuracy | 64.80 | 64.78 | -0.02 | 64.93 | +0.13 |
| LAMBADA PPL (lower is better) | 4.12 | 4.11 | improved | 4.12 | tied |

Table 17 lists only the cells of a 19-task suite where at least one of the two configurations is positive; the remaining cells are not shown.

### Loop count K (Figure 4, 16-task macro; model not named in the caption, baseline about 55.6)
- The paper's method stays above baseline for all K from 1 to 24; read off the plot it peaks at roughly 56.6 around K = 6 to 8 (approximate reading).
- The uniform loop stops helping for K ≥ 6 (about 54.4 at K = 12, approximate reading).
- The naive loop degrades monotonically and reaches 37.89% at K=6 (-17.71 pp).
- Qwen3-4B-Instruct at [17-19], 16-task Δpp (Table 16): Euler K=2 +0.89; Euler K=4 +0.73; Heun K=1 +0.15.

### Comparison with naive looping (Figure 5; naive loop uses K=4)
| Model | Benchmark | Baseline | Ours | Naive loop |
|---|---|---|---|---|
| Llama-3.2-3B-Instruct | GPQA-Main | 29.91 | 31.03 | 26.50 |
| Llama-3.2-3B-Instruct | MMLU-Pro | 31.64 | 32.36 | 27.80 |
| Llama-3.2-3B-Instruct | MMLU | 59.66 | 60.39 | 55.50 |
| Moonlight-16B-A3B-Instruct | ARC-C | 62.20 | 62.71 | 58.00 |
| Moonlight-16B-A3B-Instruct | MMLU(1500) | 67.86 | 68.60 | 63.50 |
| Moonlight-16B-A3B-Instruct | CSQA | 78.62 | 79.11 | 74.50 |

Toy example (Figure 2, tiny MLP with a 3-residual-layer block): test MSE baseline 0.015; naive K=2 2.88 vs sub-step 0.36 (about 8x); K=8 naive 335 vs sub-step 1.04 (about 320x). Note that sub-stepping is still worse than the baseline in this toy (0.36 vs 0.015).

### Ablations on Qwen3-1.7B-Base [12-15] (Table 7; Δpp on 16-task macro vs no loop)
| Axis | Setting | Δpp |
|---|---|---|
| Width (K=2 damped Euler) | n=1 | +0.18 |
| Width | n=3 | +0.21 |
| Width | n=4 | +0.55 |
| Width | n=6 | -0.82 |
| Width | n=12 | -0.63 |
| Width | n=28 (full) | catastrophic divergence |
| K and strategy (n=4) | naive loop K=2 | +0.51 |
| K and strategy | naive loop K=4 | -10.21 (0.6B) |
| K and strategy | damped Euler K=2, α=0.5 | +0.55 |
| K and strategy | damped Euler K=3, α≈1/3 | +1.05 (4B) |
| Cache (K=2 damped Euler) | first | ≥0 |
| Cache | last | ≥0 |
| Cache | none | catastrophic divergence |
| Cache detail | GSM8K, none | -58.91 |
| Cache detail | MMLU-Pro, none | -8.86 |
| Cache detail | MBPP, last minus first | +0.60 |

Higher-order solvers and accelerators vs damped Euler K=2, α=0.5 (Table 8; 16-task macro Δpp, each column vs its own reference):

| Family | Method (K=2 unless noted) | Block-mode Δ | Layer-mode Δ |
|---|---|---|---|
| Reference | damped Euler (α=0.5, h=1/2) | 0.00 | 0.00 |
| Higher-order RK | midpoint (RK2) | -0.85 | -1.96 |
| Higher-order RK | Heun (RK2) | -0.92 | -1.90 |
| Higher-order RK | RK4 | -0.65 | -2.34 |
| Fixed-point acceleration | heavy-ball (α=0.5, β=0.3) | -0.12 | -0.06 |
| Fixed-point acceleration | Anderson (K=4, m=2, β=0.5) | -0.96 | not tested |
| Fixed-point acceleration | Aitken Δ² (safeguarded) | -7.00 | not tested |
| Fixed-point acceleration | Anderson, worst (K=8, m=3, β=1.0) | -18.06 | -19.56 |

Fixed-point acceleration sweep, block-mode (Table 9; Δ vs K=2 damped Euler reference 56.113):

| Strategy | K | Hyperparameters | 16-task | Δ |
|---|---|---|---|---|
| Euler-sched | 2 | α=[0.6, 0.4] | 56.085 | -0.028 |
| heavy-ball | 2 | α=0.5, β=0.3 | 55.999 | -0.115 |
| heavy-ball | 2 | α=0.5, β=0.5 | 55.993 | -0.120 |
| heavy-ball | 2 | α=0.7, β=0.3 | 55.946 | -0.167 |
| heavy-ball | 4 | α=0.3, β=0.5 | 55.927 | -0.186 |
| Euler-sched | 4 | α=[0.7, 0.5, 0.3, 0.1] | 55.478 | -0.636 |
| Euler-sched | 3 | α=[0.7, 0.5, 0.3] | 55.252 | -0.861 |
| Anderson | 4 | m=2, β=0.5 | 55.153 | -0.960 |
| Euler-sched | 3 | α=[0.3, 0.5, 0.7] | 55.036 | -1.077 |
| Anderson | 6 | m=3, β=0.5 | 54.722 | -1.391 |
| Anderson | 3 | m=2, β=1.0 | 54.373 | -1.740 |
| Anderson | 6 | m=2, β=0.5 | 54.333 | -1.781 |
| heavy-ball | 4 | α=0.5, β=0.3 | 54.044 | -2.069 |
| Euler-sched | 4 | α=[0.4, 0.6, 0.6, 0.4] | 54.005 | -2.108 |
| heavy-ball | 4 | α=0.5, β=0.5 | 53.495 | -2.618 |
| Anderson | 4 | m=3, β=1.0 | 52.655 | -3.458 |
| heavy-ball | 4 | α=0.5, β=0.7 | 52.562 | -3.552 |
| Anderson | 4 | m=2, β=1.0 | 52.032 | -4.082 |
| heavy-ball | 3 | α=1.0, β=0.5 | 51.236 | -4.877 |
| Aitken Δ² | 2 | safeguarded | 49.109 | -7.004 |
| Aitken Δ² | 4 | safeguarded | 48.318 | -7.795 |
| Aitken Δ² | 6 | safeguarded | 44.921 | -11.192 |
| Anderson | 6 | m=2, β=1.0 | 44.261 | -11.852 |
| Anderson | 8 | m=3, β=1.0 | 38.058 | -18.056 |

Norm-stab and poly-blend sweep (Table 10; same reference, no-loop baseline 55.598):

| Strategy | K | Hyperparameters | 16-task | Δ |
|---|---|---|---|---|
| norm_stab | 2 | α=0.5 | 56.059 | -0.055 |
| poly_blend | 2 | w=[0.4, 0.2, 0.4] | 55.974 | -0.139 |
| poly_blend | 2 | w=[0.1, 0.4, 0.5] | 55.884 | -0.229 |
| poly_blend | 2 | w=[0.2, 0.3, 0.5] | 55.753 | -0.360 |
| poly_blend | 2 | w=[0.0, 0.5, 0.5] | 55.599 | -0.514 |
| poly_blend | 2 | w=[0.25, 0.25, 0.5] | 55.371 | -0.742 |
| poly_blend | 2 | w=[0.5, 0.0, 0.5] | 55.107 | -1.006 |
| norm_stab | 2 | α=0.7 | 55.009 | -1.105 |
| poly_blend | 3 | w=[0.2, 0.2, 0.2, 0.4] | 54.818 | -1.295 |
| poly_blend | 3 | w=[0.1, 0.2, 0.3, 0.4] | 54.681 | -1.433 |
| norm_stab | 3 | α=0.5 | 53.270 | -2.844 |
| norm_stab | 2 | α=1.0 | 52.302 | -3.811 |
| norm_stab | 6 | α=0.3 | 49.560 | -6.554 |
| norm_stab | 3 | α=0.7 | 49.338 | -6.776 |
| norm_stab | 4 | α=0.5 | 49.067 | -7.046 |
| norm_stab | 3 | α=1.0 | 43.101 | -13.013 |

Layer-mode strategy sweep (Table 11; Δ vs layer-mode K=2 damped Euler, reference 56.347):

| Recipe | 16-task | Δ |
|---|---|---|
| K=2 heavy-ball α=0.5, β=0.3 | 56.292 | -0.056 |
| K=2 poly_blend w=[0.25, 0.5, 0.25] | 56.181 | -0.167 |
| K=3 damped Euler α=0.3 | 55.278 | -1.069 |
| K=4 Euler | 55.240 | -1.107 |
| K=2 damped Euler α=0.7 | 55.196 | -1.151 |
| K=6 Euler | 54.998 | -1.350 |
| K=2 Heun | 54.445 | -1.902 |
| K=2 midpoint | 54.385 | -1.962 |
| K=1 RK4 | 54.242 | -2.105 |
| K=1 midpoint | 54.178 | -2.169 |
| K=2 norm_stab α=0.5 | 54.098 | -2.249 |
| K=1 Heun | 54.060 | -2.287 |
| K=2 RK4 | 54.005 | -2.342 |
| K=3 heavy-ball α=0.5, β=0.3 | 51.830 | -4.517 |
| K=3 Anderson m=2, β=1.0 | 44.035 | -12.312 |
| K=4 Anderson m=2, β=1.0 | 36.785 | -19.562 |

Layer-mode on dense backbones (Table 18):

| Model | Configuration | 16-task Δpp | MMLU 5sh Δpp |
|---|---|---|---|
| Qwen3-0.6B-Base | layer K=2 heavy-ball β=0.5 [12-15] | +0.561 | +0.128 |
| Qwen3-0.6B-Base | layer K=2 heavy-ball β=0.3 [12-15] | +0.442 | +0.214 |
| Qwen3-0.6B-Base | layer K=2 poly-blend [0.25, 0.5, 0.25] [12-15] | +0.236 | not given |
| Qwen3-1.7B-Base | layer K=2 damped Euler [12-15] | +0.234 | -0.477 (does not generalize) |
| Qwen3-1.7B-Base | layer K=2 heavy-ball [12-15] | +0.178 | not given |
| Llama-3.2-3B-Inst | layer K=2 Euler [16-19] | not given | +0.72 |

### Cache strategy (Table 20; Qwen3-4B-Instruct [15-18] K=3 Euler, Δpp)
| Cell | cache=first | cache=last | cache=none |
|---|---|---|---|
| MMLU-Pro 5-shot CoT | +2.64 | +1.00 | -8.86 |
| MBPP 3-shot pass@1 | +0.20 | +0.80 | -23.40 |

### Per-subject concentration (App. H)
MMLU 5-shot, Qwen3-4B-Base, [17-19] K=2 Euler (Table 13). The 57-subject macro is roughly unchanged (73.01 -> 72.90, -0.11 pp):

| Subject | Baseline | Loop | Δpp |
|---|---|---|---|
| college_physics | 62.75 | 68.63 | +5.88 |
| high_school_mathematics | 56.30 | 61.85 | +5.56 |
| abstract_algebra | 55.00 | 59.00 | +4.00 |
| jurisprudence | 79.63 | 83.33 | +3.70 |
| high_school_statistics | 75.46 | 77.78 | +2.31 |
| conceptual_physics | not given | not given | +2.13 |
| us_foreign_policy (K=1 Heun) | not given | not given | +2.00 |

MMLU 0-shot, Qwen3-4B-Instruct, [17-19] K=2 Euler (Table 14; subjects with ≥ +3 pp). Macro over 57 subjects +0.99 pp (68.15 -> 69.14):

| Subject | Baseline | Loop | Δpp |
|---|---|---|---|
| global_facts | 32.00 | 38.00 | +6.00 |
| high_school_physics | 60.26 | 65.56 | +5.30 |
| high_school_mathematics | 47.41 | 52.59 | +5.18 |
| econometrics | 62.28 | 65.79 | +3.51 |
| medical_genetics | 75.00 | 79.00 | +4.00 |
| professional_medicine | 72.43 | 76.47 | +4.04 |
| security_studies | 70.20 | 73.88 | +3.68 |

MMLU-Redux 30-subject sweep on Qwen3-4B-Instruct: 14 of 30 subjects positive, 10 tied, 6 mildly negative; +0.93 pp macro; largest gains professional_accounting (+8.00), high_school_physics (+7.00), econometrics (+5.00), college_chemistry (+3.00/+4.00 under K=1 Heun).

### Medical or domain-specific benchmarks
- MedMCQA 0-shot, Qwen3-4B-Instruct: 53.40 -> 54.13 (+0.73 pp) at [11-14] K=3 Euler, cache first, block, decode full. This is the only medical benchmark, on one model, and its window differs from that model's usual [15-18].
- MMLU 0-shot medical subjects on Qwen3-4B-Instruct ([17-19] K=2 Euler): medical_genetics 75.00 -> 79.00 (+4.00), professional_medicine 72.43 -> 76.47 (+4.04). The paper itself warns that per-subject MMLU sets of 100-300 examples produce false signal.
- MedQA: not evaluated. Medical fine-tunes: not evaluated. Phi-3.5-mini / MediPhi: not evaluated.
- Other domain sets: C-Eval (Chinese) +0.52; MBPP (code) only in cache ablations; GSM8K only in wall-clock and cache ablations.

### Negative results (where looping hurts)
- Naive looping: 37.89% at K=6 (-17.71 pp, Figure 4b). Naive K=4 on Qwen3-0.6B-Instruct raised LAMBADA perplexity from about 13 to 1054.12 and cut the 16-task macro by 10.21 pp. Naive K=4 loses about 3.4 to 4.4 pp against baseline on all six cells of Figure 5 (computed from the figure values).
- Wide windows: n=6 -0.82 pp, n=12 -0.63 pp; n=28 (whole network) drives LAMBADA perplexity to 6.3 x 10^5.
- Higher-order solvers: block-mode losses of 0.7 to 1.1 pp, layer-mode 1.7 to 2.3 pp. Anderson m=3, β=1.0, K=8: -18.06 pp, the worst single configuration reported. norm_stab fails badly at K ≥ 3.
- Layer-mode with K=3: heavy-ball on Qwen3-4B-Base [15-18] loses 5.0 pp on 16-task and 6.0 pp on MMLU 5-shot; layer-mode is said to tolerate only K=2.
- cache=none: MBPP 3-shot pass@1 38.20 vs 61.60 (-23.40 pp); MMLU-Pro 5-shot 48.29 vs 57.14 (-8.86 pp); GSM8K -58.91.
- Small distilled models: Llama-3.2-1B MMLU (-0.63) and MMLU-Pro (-1.36) stay negative even after per-cell tuning; the authors put the boundary below about 1.7B for Qwen3 and about 3B for Llama.
- Position search overfits: Qwen3-0.6B-Base [8-11] +1.33 on 16-task but -1.21 on MMLU 5sh; Qwen3-1.7B-Base [7-10] +0.92 on 16-task but -1.38 on MMLU 5sh.
- Qwen3-1.7B-Base layer K=2 damped Euler: +0.234 on 16-task, -0.477 on MMLU 5sh.
- MoE block-mode: 7 of 12 block-mode cells in Table 2c are negative (count from the table), down to -2.00 (Moonlight GPQA-Main).
- Leakage-free 30B run: GPQA-Diamond -1.51 under Euler K=2.
- Uniform loop of [[Lys et al. 2026 - Inner Loop Inference]] falls below baseline for larger K in Figure 4a.

### Compute overhead
Wall-clock, Qwen3-4B-Instruct, GSM8K@200, K=3 Euler at [15-18] (Table 15):

| decode_mode | seconds | overhead vs baseline |
|---|---|---|
| no loop (baseline) | 194 | - |
| bypass | 191 | -1.5% |
| first_n, N=16 | 192 | -1.0% |
| first_n, N=64 | 203 | +4.6% |
| full | 236 | +21.6% |

Memory and suite wall-clock (Table 12; full 16-task suite; CoT generation jobs take 2-5x longer in decode_mode=full):

| Model | VRAM (bf16) | Hardware used | Eval-suite wallclock |
|---|---|---|---|
| Qwen3-0.6B-Base / -Instruct | ~2 GB | A100-40 / H100-80 | ~1 h |
| Qwen3-1.7B-Base | ~4 GB | A100-40 / H100-80 | ~2 h |
| Qwen3-4B-Base / -Instruct | ~8 GB | A100-40 / H100-80 | ~3 h |
| Llama-3.2-1B / -3B-Instruct | ~3-7 GB | A100-40 / H100-80 | ~2 h |
| Qwen1.5-MoE-A2.7B-Chat | ~30 GB | H100-80 | ~4 h |
| DeepSeek-V2-Lite-Chat (16B/2.4B) | ~32 GB | H100-80 | ~5 h |
| Moonlight-16B-A3B-Instruct | ~32 GB | H100-80 | ~8 h (decode=full 21 h) |
| Qwen3-30B-A3B-Instruct | ~60 GB | H100-80 | ~10 h (19-task 0-shot) |

LAMBADA perplexity is preserved or improved: Qwen3-4B-Base [17-19] K=2 Euler 4.25 -> 4.19; Qwen3-4B-Instruct 7.30 -> 7.01; Qwen3-30B-A3B-Instruct [22-24] 4.12 -> 4.11.

### Inconsistencies inside the source (checked against the PDF)
- Tuning: the contributions and Table 3 caption claim no per-cell hyperparameter tuning, but App. P says the 45 cells come from a structured per-cell search, Section 3.4 says Table 3 reports the best configuration per model family, Table 6 gives different windows for different benchmarks of the same model, and App. G.8 mentions cells that stayed negative after per-cell tuning.
- Recipe naming: "K-stage Runge-Kutta" (contributions), "3-stage" (Section 3.1), "K=2-stage" (App. D), while Table 6 labels the runs "Euler" or "damped Euler". β is not reported for any headline cell.
- MoE mode: the recipe says layer-mode for MoE, but all Qwen1.5-MoE rows and several DeepSeek / Moonlight rows in Table 6 are block-mode, and Qwen3-30B-A3B uses block-mode.
- Qwen1.5-MoE ARC-Challenge: +2.30 with block K=3 at [13-16] in Tables 3 and 6, but +0.17 for block K=3 at [13, 16] in Table 2c.
- Llama-3.2-3B GPQA-Main: +1.12 in Table 3 and Figure 5, +0.67 in Table 6 and App. J.5. Llama-3.2-3B MMLU: 0-shot in Table 3, 5-shot in Table 6.
- Qwen3-4B-Base MMLU 5-shot baseline: 73.27 (Table 6) vs 73.01 (Table 13 caption, which also gives +0.60 instead of +0.34).
- Cache default: App. N says all numbers use c = LAST unless stated otherwise; Table 6 is mostly `first`; App. D calls `cache=first` the robust default.
- Decode default: App. I says bypass is the default for the log-likelihood benchmarks behind the headline gains, while the headline Qwen3-4B-Instruct rows in Table 6 are decode `full`.
- Naive K=2: Table 7b reports +0.51 (close to damped Euler +0.55), while Figure 4b shows the naive curve below baseline at K=2 and the text says naive looping degrades performance almost universally.
- Figure 4 caption cites Algorithm 5 for the paper's method, which is the layer-mode decode algorithm; Algorithm 3 is probably meant (inference).
- Counts: seven model families, eight tested architectures, nine architectures are all used. Qwen3-1.7B-Base 16-task baseline is 55.54 by back-calculation from Table 2a, 55.60 in Table 6, 55.598 in Table 10.

## What it establishes
- On the tested checkpoints, undamped reapplication of a mid block degrades accuracy at K ≥ 4 (Figure 4b, Figure 5, App. G.1), and whole-network or wide-window looping collapses.
- A damped update with step about $1/K$ on a 4-layer mid-depth window keeps the model in a usable regime up to K = 24 on the one model shown in Figure 4, and preserves LAMBADA perplexity on the three reported configurations.
- With per-cell selected configurations, reported gains are between +0.30 and +2.64 pp (Table 6); with a pre-committed configuration on Qwen3-30B-A3B-Instruct they are between -1.51 and +1.14 pp for Euler K=2 (Table 17).
- Higher-order ODE solvers and fixed-point accelerators (Heun, midpoint, RK4, Anderson, Aitken, heavy-ball) do not beat damped Euler K=2 on Qwen3-1.7B-Base [12-15].
- For autoregressive decoding with looped layers, the loop region must write a KV entry; writing none is catastrophic, and a snapshot/crop plus single stash pass keeps the cache shape identical to the unmodified model.
- Good windows are broad in position: many 4-layer windows are positive on the 16-task macro (Table 19).

## What it does NOT establish
- Statistical significance: no confidence intervals, no paired tests, point estimates only. For benchmarks with ≤1,000 examples a +2 pp gain is at most about 20 questions, and the unpaired binomial standard error at n = 1,000 and accuracy 0.34 is about 1.5 pp (inference).
- That the reported per-cell gains are free of selection: the per-cell best configuration is chosen on the benchmark it is reported on, and the only held-out check is a different benchmark (MMLU 5-shot) (inference from App. P and Table 6).
- That damping is needed at K=2: the paper's own Table 7b has naive K=2 at +0.51 vs damped +0.55.
- Anything about MedQA, Phi-3.5-mini, MediPhi, or any medical fine-tune; the medical evidence is one MedMCQA cell (+0.73 pp) and two small MMLU subjects.
- That different domains need different blocks: Table 6 shows benchmark-dependent best windows, but the paper attributes position-search differences of about 1 pp to overfitting (App. G.5) and does not test domain-specific windows.
- That the ODE interpretation is correct. The authors themselves write that "the looped block is not a smooth ODE" (App. D) and that it is not usefully contractive; in the toy example sub-stepping is worse than the baseline.
- Benefit for long chain-of-thought generation in general: GSM8K appears only for wall-clock and the cache=none failure; no GSM8K accuracy gain is reported.
- Reproducibility from released code: no repository is stated, and the 16-task and 19-task suites are not listed.

## Limitations stated by the authors
There is no dedicated limitations section. Caveats stated in the text:
- The 16-task suite is not a reliable target for claims below 1 pp, because small MMLU subjects (100-300 examples) create false signal.
- Position search can overfit to the screen (App. G.5, J.1).
- Failures concentrate in sub-3B distilled checkpoints on knowledge multiple-choice tasks; such models are said to lack the mid-layer redundancy that looping relies on.
- MLA-based MoE models gain 3-4x less.
- Layer-mode tolerates only K=2; block-mode on MoE suffers routing thrash.
- Wide windows and whole-network looping fail; the usable region is described as a contiguous mid-band of about 4 layers.
- Point estimates only; abs(Δ) ≤ 0.3 pp on small benchmarks is treated as neutral because run-to-run variance is up to 0.5 pp there.
- Headline results concern knowledge-heavy multiple-choice tasks with baselines below ceiling, scored with a lenient answer extractor.

## Implementation notes for med_wsp
- Code: no public link stated. We implement it ourselves; the paper patches the top-level decoder forward (the layer loop), not individual layers.
- Minimal rule to implement (block-mode, eq. 10 / 16): compute `anchor = g(x)`, then K damped steps `x <- x + (g(x) - x) / K` (reuse `anchor` for the first), then `x <- beta * anchor + (1 - beta) * x`. Expose `alpha` separately from `1/K` so that naive (`alpha = 1`) and schedules share one code path (inference).
- Hook-based variant (inference): a forward pre-hook on layer `a` can capture the window input and layer kwargs, and a forward hook on layer `b` can re-run layers `a..b` and return the damped state. This is enough when no KV cache is used.
- MedQA as multiple-choice scoring needs only a prefill forward pass per prompt, so the KV-cache protocol (Algorithm 4) can be skipped by running with `use_cache=False` (inference; this corresponds to the paper's `bypass` regime).
- If we generate chain-of-thought: implement snapshot/crop per iteration plus one stash pass; `cache=first` for long CoT, `cache=last` for short structured output, never `none`. Check that the installed transformers cache class still appends on `update()` and supports cropping (the paper used transformers 4.46).
- Defaults transferred to a 32-layer dense model (inference, not tested in the paper): block-mode; n=4; window center in 0.45-0.60 means starts a = 13 to 17, with [14-17] (center 0.484) the closest analogue of Qwen3-4B [15-18] (0.458) and Qwen3-1.7B [12-15] (0.482); K=2 or 3; avoid n ≥ 6; avoid layer-mode with K ≥ 3.
- Cost (inference from the paper's formula): extra window passes are $(b-a+1)(K-1)$, so n=4 on 32 layers adds 4/32 = 12.5% layer compute at K=2 and 25% at K=3 for a prefill-only evaluation. The paper measured +21.6% wall-clock for K=3, n=4, decode `full` on a 36-layer model.
- Fits on a 36 GB M4 Max (inference): Phi-3.5-mini (3.8B) and MediPhi at 16-bit, comparable to the ~8 GB the paper reports for Qwen3-4B. bfloat16 was used on CUDA; dtype support on MPS needs checking. Does not fit: the ~60 GB Qwen3-30B-A3B run; the ~30-32 GB MoE models would be marginal. The authors' total budget (about 20,000 H100 hours, about 720 configurations) is far beyond one laptop, so our sweep must be much smaller.
- Gotcha: validate any chosen window on questions not used for selection; the authors saw +0.9 to +1.3 pp screen gains turn into -1.0 to -1.4 pp on held-out data.

## Relevance to med_wsp (provisional)
- Primary method source for the "Training-Free Reasoning" experiment on Phi-3.5-mini and [[Corbeil et al. 2025 - MediPhi]]; see [[Layer looping]] for the side-by-side with [[Lys et al. 2026 - Inner Loop Inference]].
- Expected effect size on a medical MC benchmark, if the MedMCQA cell transfers: under +1 pp (provisional reading of one cell). This sets the bar for how many paired MedQA questions we need.
- The base-vs-instruct comparisons here are not a base-vs-domain-fine-tune comparison; whether MediPhi prefers a different window than Phi-3.5-mini is untested by this paper (provisional question for us).
- For an agent such as GPT-6 Astra proposing layers: the paper's own search space (center, width, K, strategy, cache) and the depth-fraction default give a concrete non-agent baseline to compare against (provisional idea).
- Phi-3.5-mini (3.8B) is above the sizes where the paper reports failures (below about 1.7B for Qwen3, below about 3B for Llama), but whether that boundary transfers to the Phi family is unknown (inference).

## Open questions
- Which $\beta$ did the headline runs use, and does $\beta > 0$ matter at K = 2 or 3?
- Is the +0.51 naive K=2 result real, that is, is damping only necessary for K ≥ 3?
- What are the 16-task and 19-task suites?
- Does the depth-fraction rule hold for Phi-3.5-mini (32 layers) and for a medical fine-tune of it?
- Are benchmark-specific best windows (for example [11-14] for MedMCQA vs [15-18] elsewhere) reproducible on held-out questions, or selection noise?
- What do "EMA" and "halt τ=0.05" mean?

## Ledger card
**Full citation:** Lizhang Chen, Jonathan Li, Chen Liang, Ni Lao, Qiang Liu. "Training-Free Looped Transformers." arXiv:2605.23872v1 [cs.LG], 22 May 2026. Preprint. (Chen and Li marked as equal contribution; University of Texas at Austin.)
**Source / DOI / URL:** arXiv:2605.23872; local text `papers/pdf/text/arxiv_2605.23872.txt`, PDF `papers/pdf/arxiv_2605.23872.pdf` (34 pages).
**Why I read it:** Method source for looping layers of a frozen model at inference time; need the exact update rule, block selection and KV handling.
**Problem addressed:** Whether a frozen off-the-shelf checkpoint can be looped at inference with no training, parameters or architecture change.
**Method / representation:** Re-apply a contiguous mid window $[a,b]$; damped sub-steps $x_{k+1} = (1-\tfrac1K)x_k + \tfrac1K g(x_k)$; optional anchor blend $\beta g(x_0) + (1-\beta)F^K(x_0)$; block-mode or layer-mode; KV snapshot/crop plus stash pass.
**Assumptions:** Pre-norm residual blocks read as forward-Euler steps of $\dot x = F_g(x)$; mid-layer redundancy; post-loop layers expect the $t=1$ endpoint.
**Evidence used by the paper:** experiments: 45 (model, benchmark) cells, ablations on Qwen3-1.7B-Base [12-15], a pre-committed transfer to Qwen3-30B-A3B-Instruct / analysis: ODE and Runge-Kutta derivation, two short proofs, toy MLP / datasets or systems: Qwen3 0.6B-4B and 30B-A3B, Qwen1.5-MoE-A2.7B, Llama-3.2 1B/3B, DeepSeek-V2-Lite, Moonlight-16B-A3B; MMLU, MMLU-Pro, GPQA, CSQA, ARC, OBQA, SciQ, MedMCQA, C-Eval and others.
**Safe citation claims:** "Chen et al. (2026) report that naive reapplication of a mid-layer block of a frozen LLM degrades accuracy and propose damped sub-steps with step 1/K"; "they report gains of up to +2.64 pp (MMLU-Pro, Qwen3-4B-Instruct) with per-cell selected configurations and +0.73 pp on MedMCQA"; "they report that a 4-layer window near depth fraction 0.45-0.60 works best on their checkpoints".
**Unsafe / unsupported citation uses:** "looping improves medical reasoning"; "gains are statistically significant"; "no hyperparameter tuning was needed" (contradicted by App. P); "looping a transformer solves an ODE more accurately" (the authors find higher-order solvers hurt); any claim about Phi-3.5, MediPhi or MedQA.
**Relation to our project:** closest prior art | overlapping method
**Verification status:** full text read: yes (including appendices A-Q); key claim checked: yes for formulas and tables against the PDF (Algorithm 3, Tables 1-3, 6, 7, App. P, Figures 3-6); experimental claims not reproduced.

## Links
- [[Layer looping]]
- [[Lys et al. 2026 - Inner Loop Inference]]
- [[Corbeil et al. 2025 - MediPhi]]
- [[Activation patching]] (hook-based interventions on the residual stream, related tooling)
