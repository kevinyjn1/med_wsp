---
type: concept
title: "Layer looping"
tags: [concept, layer-looping, training-free, inference-time]
---

# Layer looping

Re-running a contiguous block of layers of a frozen transformer at inference, with no weight change. Sources: [[Chen et al. 2026 - Training-Free Looped Transformers]] (Chen) and [[Lys et al. 2026 - Inner Loop Inference]] (Lys). Both find naive reapplication $x \leftarrow g(x)$ harmful and add damping or averaging.

## Update rules
`run(i, j, h)`: ordinary forward through layers i..j inclusive (attention and MLP recomputed on `h`).

```python
# Chen: K damped sub-steps on window [a, b] (block-mode, eqs. 10, 16)
def loop_chen(x, a, b, K, beta=0.0):
    anchor = run(a, b, x)                    # g(x0)
    x = x + (anchor - x) / K                 # sub-step 1
    for _ in range(K - 1):
        x = x + (run(a, b, x) - x) / K
    return beta * anchor + (1 - beta) * x    # then layers b+1..

# Lys: R applications of blocks s..e-1 with interpolation (eqs. 6-12)
def loop_lys(h0, s, e, R, mode, eta):
    cache, h_hat = [h0], h0
    for t in range(1, R + 1):
        cache.append(run(s, e - 1, h_hat))   # h(t) = F(h_hat(t-1))
        if mode == "naive":   h_hat = cache[-1]
        if mode == "uniform": h_hat = mean(cache)
        if mode == "moving":  h_hat = eta * h0 + (1 - eta) * cache[-1]
        if mode == "auto":    h_hat = mix(softmax([dot(h0, h) for h in cache]), cache)
    return h_hat                             # then blocks e..
```

- Chen: $x_{k+1} = (1-\tfrac1K)x_k + \tfrac1K g(x_k)$; output $\beta g(x_0) + (1-\beta)F^K(x_0)$. The printed Algorithm 3 runs $K-1$ sub-steps after the anchor; the code follows eq. 16 (inference). Layer-mode applies the rule to each layer in turn (for MoE).
- Lys: $\hat h^{(t)} = \sum_i \alpha_i h^{(i)}$, $\sum_i \alpha_i = 1$. Taking $h^{(0)}$ as the loop input and returning $\hat h^{(R)}$ is my reading (inference).

## Hyperparameters to sweep
| Knob | Chen | Lys |
|---|---|---|
| Block start | $a$ | $s$ |
| Block length | $n=b-a+1$; 4 best, 6 or more hurts | $N=e-s$; all intervals swept |
| Loops | $K$ = 2 or 3 | $R$; 3 in the naive sweep, otherwise not stated |
| Damping | step $\alpha=1/K$; anchor $\beta$ (headline value not stated) | uniform; $\eta$ (not stated); softmax |
| Other | block vs layer mode; cache first/last; decode bypass/full | matched-noise control |

## How the block is chosen
- Chen: 4 layers centered at depth fraction 0.45-0.60. Reported per-cell results also come from a search (center in $[0.40N, 0.70N]$, width 3/4/6, $K$ 2/3), screened on a 16-task suite and re-checked on MMLU 5-shot; only the Qwen3-30B run was pre-committed.
- Lys: full start-end sweep on WinoGrande with the uniform rule, then frozen for other benchmarks (Gemma 2-2B: layers 10-13; others about 40-60% of depth). No separate validation split.

## KV cache and generation
- Chen: prefill loops with `use_cache=False`, then one stash pass writes the window's KV. Decode: each iteration reads past KV and the cache is cropped back; a stash pass then writes one entry per layer from the pre-loop (`first`) or post-loop (`last`) state. Writing none is catastrophic; `bypass` loops only in prefill.
- Lys: not stated.
- Single-pass multiple-choice scoring needs no cache (inference).

## Sweep plan for Phi-3.5-mini, 32 layers, MedQA (provisional design)
1. Baselines for Phi-3.5-mini and [[Corbeil et al. 2025 - MediPhi]]; single-pass answer scoring; fixed selection and test splits.
2. Defaults, no search: Chen block-mode $[14,17]$ (center 0.48), $K\in\{2,3\}$, $\beta=0$; Lys uniform on the same blocks, $R=3$.
3. Position scan: 4-layer windows, starts 2, 4, ..., 26, Chen $K=2$; add naive $K=2$ and a noise control.
4. Around the best start: length $\{2,3,4,6\}$, $K\in\{2,3,4\}$, $\beta\in\{0,0.5\}$; Lys modes with $R\in\{2,3\}$, $\eta\in\{0.25,0.5,0.75\}$.
5. Domain contrast: repeat step 3 on a non-medical set and on MediPhi; compare best windows on held-out questions.
6. Statistics: paired McNemar. Questions needed $\approx 7.85\,d/\Delta^2$ (80% power, two-sided 5%; $d$ = fraction of questions whose correctness flips): $d=0.05$ gives about 980 for 2 pp and 3,900 for 1 pp (inference). Sources report mostly under +1 pp, at most +2.64 pp.
