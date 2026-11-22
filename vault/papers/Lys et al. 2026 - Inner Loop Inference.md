---
type: paper
title: "Inner Loop Inference for Pretrained Transformers: Unlocking Latent Capabilities Without Training"
short: "Lys et al., 2026"
authors: [Jonathan Lys, Vincent Gripon, Bastien Pasdeloup, Axel Marmoret, Lukas Mauch, Fabien Cardinaux, Ghouthi Boukli Hacene]
year: 2026
venue: "arXiv:2602.14759v2 [cs.LG], 2 Mar 2026 (publication venue not stated)"
url: "https://arxiv.org/abs/2602.14759"
local_text: "papers/pdf/text/arxiv_2602.14759.txt"
tags: [paper, layer-looping, training-free, inference-time, representation-interpolation]
read_status: full-text-read
---

# Lys et al., 2026 - Inner Loop Inference

## TL;DR
The paper re-applies a contiguous range of blocks $\{s, \dots, e-1\}$ of a frozen language model $R$ times at inference ("inner looping") and stabilises the result by interpolating each looped hidden state with a cache of earlier loop states: $\hat h^{(t)} = \sum_{i=0}^{t} \alpha_i h^{(i)}$ with $\sum_i \alpha_i = 1$. Three weightings are tested (uniform average, moving average against the baseline state, softmax "auto-alignment") on Gemma 2 (2B, 9B) and Llama 3-8B. Naive looping with a loop count of 3 is worse than baseline for every start-end pair on Gemma2-2B WinoGrande. The block is picked by sweeping all intervals on WinoGrande with the uniform rule (Gemma 2-2B: layers 10-13; other models roughly 40-60% of depth) and then kept fixed. Gains are small: for example Gemma-2B WinoGrande 68.75 ± 1.30 -> 69.53 ± 1.29 (Uniform) and GSM8K 25.02 ± 1.19 -> 26.38 ± 1.21 (Moving Average); Llama 3-8B is mixed (MMLU 34.50 -> 33.95 under Uniform).

## Problem
If depth mainly sets how many refinement steps are applied to a shared latent state (the "logits refinement" view supported by logit-lens and layer-robustness findings), can extra refinement steps be added at inference time to improve a frozen model, without changing parameters? The authors frame looping as a test-time probe of refinement dynamics in pretrained models, in contrast to work that trains looped or latent-reasoning architectures.

## Method (step by step)

### 1. Notation
- Embedding (eq. 1): $h_0 = (E[x_1], \dots, E[x_T])$, $E \in \mathbb{R}^{|V| \times d}$.
- Blocks (eq. 2): $h_\ell = B_\ell(h_{\ell-1})$, $\ell = 1, \dots, L$.
- Each block (eqs. 3-4): $z = h_{\ell-1} + \mathrm{MSA}_\ell(\mathrm{LN}(h_{\ell-1}))$, $h_\ell = z + \mathrm{FFN}_\ell(\mathrm{LN}(z))$. Gemma additionally applies post-normalization layers.

### 2. Which blocks are re-applied, and how many times
- Middle-looping scheme with three parameters: $s$ = index of the first block in the loop, $e$ = index of the block following the loop, $R$ = number of repetitions.
- Blocks before $s$ run once; blocks in $\{s, s+1, \dots, e-1\}$ are repeated $R$ times; blocks from $e$ onward run once.
- Loop length $N = e - s$; total block applications $K = L + (R-1)N$.
- Block mapping (eq. 5):

$$\pi(k) = \begin{cases} k & k < s \\ s + (k - s) \bmod N & s \le k < s + RN \\ k - (R-1)N & k \ge s + RN \end{cases}$$

- The whole range is iterated as a unit (block 1 to block $N$ of the range, then again), which corresponds to block-mode in [[Chen et al. 2026 - Training-Free Looped Transformers]]. Each application runs the full blocks (attention and FFN) on the current state (inference from eq. 6: $F$ is one full application of the block range; no reuse of attention across loop steps is described).
- Indexing note: the text gives $\pi$ once on $\{0..K-1\} \to \{0..L-1\}$ and once on $\{1..K\} \to \{1..L\}$.

### 3. Regularized looping (the interpolation rule)
- $h^{(0)}$: the baseline hidden state at the loop boundary from a standard forward pass. $F$: one application of the loop block range.
- Loop step $t \ge 1$, unregularized output (eq. 6): $h^{(t)} = F(\hat h^{(t-1)})$.
- Regularized state from the cache $\{h^{(i)}\}_{i=0}^{t}$ (eq. 7):

$$\hat h^{(t)} = \sum_{i=0}^{t} \alpha_i h^{(i)}, \qquad \sum_{i=0}^{t} \alpha_i = 1$$

- Variants differ only in $\alpha_i$:
  - Naive (eq. 8): $\alpha_t = 1$, $\alpha_{i<t} = 0$, so $h^{(t)} = F(h^{(t-1)})$ for $t = 1, \dots, R$.
  - Uniform (eq. 9): $\hat h^{(t)} = \frac{1}{t+1} \sum_{i=0}^{t} h^{(i)}$.
  - Moving average (eq. 10): $\hat h^{(t)} = \eta \cdot h^{(0)} + (1-\eta) \cdot h^{(t)}$.
  - Auto-alignment (eqs. 11-12): $s^{(t)} = [\langle h^{(0)}, h^{(0)} \rangle, \dots, \langle h^{(0)}, h^{(t)} \rangle] \in \mathbb{R}^{t+1}$, $\alpha^{(t)} = \mathrm{softmax}(s^{(t)})$, $\hat h^{(t)} = \sum_{i=0}^{t} \alpha^{(t)}_i h^{(i)}$.
- Noise control ("Noise" column): the loop-induced signal is replaced by random noise of matched magnitude. Distribution and how the magnitude is matched: not stated.

### 4. Things the text leaves open
- Whether $h^{(0)}$ is the state entering block $s$ or the state after one ordinary pass through the range. The naive recursion ($t = 1..R$ with $R$ total repetitions and $K = L + (R-1)N$) is consistent only with $h^{(0)}$ being the state entering block $s$ (inference). Under that reading the uniform rule at $t = 1$ already averages the block input with the block output.
- Which state is handed to block $e$ after the last step: presumably $\hat h^{(R)}$ (inference); not stated. Chen et al. render the uniform rule as $x_{k+1} = g(\frac{1}{k+1}\sum_{i=0}^{k} x_i)$, where the final state is an un-averaged block output, so the two write-ups may differ in the last step.
- $R$ for the main results: not stated (only the naive sweep states a loop count of 3).
- $\eta$ for the moving average: not stated.
- Whether the inner product in eq. 11 is per token or over the whole sequence, and any softmax temperature: not stated.
- The linked repository (github.com/jonathanlys01/looped-transformer) would settle these; it was not consulted for this note.

### 5. Block selection
- For each model, sweep all candidate start-end intervals on WinoGrande using the uniform rule.
- Gemma 2-2B: best configuration is layers 10-13 (Figure 1 heatmap; baseline 0.6875; color scale from about -0.020 to +0.010). Whether "10-13" means $s=10, e=13$ (3 blocks) or blocks 10 through 13 inclusive: not stated. The heatmap axes (start layer 0-24, end layer 2-26) suggest the end index is exclusive (inference).
- Other models: selected intervals fall at roughly 40-60% of the network depth; exact intervals for Gemma 2-9B and Llama 3-8B: not stated.
- The model-specific region is then kept fixed for all other benchmarks (no per-task re-optimization).
- Validation set: no separate validation split is described. The interval is selected on the WinoGrande evaluation itself, so the WinoGrande row of Table I is in-sample for the selection (inference); the other benchmarks are held-out with respect to the interval choice.

### 6. Evaluation protocol
- lighteval library (built on HELM).
- Multiple-choice scoring: cumulative probability of each option; correct if the true answer has the highest length-normalized likelihood. GSM8K is generative.
- Shots: WinoGrande 5, ARC Easy 0, ARC Challenge 25, GSM8K 5, HellaSwag 10, MMLU 5.

## Experiments and numbers

### Models, benchmarks, baselines
- Models: Gemma 2-2B and Gemma 2-9B (pre- and post-normalization in each block), Llama 3-8B (standard pre-norm).
- Benchmarks: WinoGrande, ARC Easy, ARC Challenge, GSM8K, HellaSwag, MMLU.
- Baselines and controls: unmodified model (Baseline), naive looping, matched-magnitude noise.

### Table I (accuracy ± reported uncertainty, quoted exactly; what the ± denotes is not stated)
| Benchmark (shots) | Model | Size | Baseline | Uniform | Moving Average | Auto-Align | Noise |
|---|---|---|---|---|---|---|---|
| WinoGrande (5) | Gemma | 2B | 68.75 ± 1.30 | 69.53 ± 1.29 | 69.06 ± 1.30 | 68.98 ± 1.30 | 68.51 ± 1.31 |
| WinoGrande (5) | Gemma | 9B | 77.35 ± 1.18 | 77.43 ± 1.17 | 77.35 ± 1.18 | 77.35 ± 1.18 | 77.35 ± 1.18 |
| WinoGrande (5) | Llama | 8B | 76.16 ± 1.20 | 76.64 ± 1.19 | 76.40 ± 1.19 | 76.09 ± 1.20 | 75.93 ± 1.20 |
| Arc E (0) | Gemma | 2B | 80.30 ± 0.82 | 80.01 ± 0.82 | 80.18 ± 0.82 | 80.22 ± 0.82 | 79.84 ± 0.82 |
| Arc E (0) | Gemma | 9B | 87.84 ± 0.67 | 87.84 ± 0.67 | 87.88 ± 0.67 | 87.84 ± 0.67 | 87.84 ± 0.67 |
| Arc E (0) | Llama | 8B | 77.69 ± 0.85 | 77.31 ± 0.86 | 77.40 ± 0.86 | 77.61 ± 0.86 | 77.57 ± 0.86 |
| Arc C (25) | Gemma | 2B | 52.82 ± 1.46 | 53.41 ± 1.46 | 52.90 ± 1.46 | 52.99 ± 1.46 | 52.65 ± 1.46 |
| Arc C (25) | Gemma | 9B | 67.75 ± 1.37 | 68.09 ± 1.36 | 68.09 ± 1.36 | 67.58 ± 1.37 | 67.49 ± 1.37 |
| Arc C (25) | Llama | 8B | 59.73 ± 1.43 | 59.64 ± 1.43 | 59.73 ± 1.43 | 60.07 ± 1.43 | 59.64 ± 1.43 |
| GSM8K (5) | Gemma | 2B | 25.02 ± 1.19 | 26.00 ± 1.21 | 26.38 ± 1.21 | 26.08 ± 1.21 | 24.79 ± 1.19 |
| GSM8K (5) | Gemma | 9B | 68.46 ± 1.28 | 68.52 ± 1.27 | 68.39 ± 1.28 | 68.31 ± 1.28 | 68.33 ± 1.28 |
| HellaSwag (10) | Gemma | 2B | 74.51 ± 0.43 | 74.70 ± 0.43 | 74.64 ± 0.43 | 74.60 ± 0.43 | 74.50 ± 0.43 |
| HellaSwag (10) | Gemma | 9B | 82.41 ± 0.38 | 82.53 ± 0.38 | 82.42 ± 0.38 | 82.42 ± 0.38 | 82.42 ± 0.38 |
| HellaSwag (10) | Llama | 8B | 82.14 ± 0.38 | 82.30 ± 0.38 | 82.31 ± 0.38 | 82.31 ± 0.38 | 82.14 ± 0.38 |
| MMLU (5) | Gemma | 2B | 53.93 ± 3.57 | 54.49 ± 3.58 | 54.35 ± 3.57 | 54.13 ± 3.57 | 53.97 ± 3.58 |
| MMLU (5) | Gemma | 9B | 72.17 ± 3.11 | 72.42 ± 3.10 | 72.23 ± 3.11 | 72.18 ± 3.11 | 72.20 ± 3.11 |
| MMLU (5) | Llama | 8B | 34.50 ± 3.47 | 33.95 ± 3.45 | 33.99 ± 3.45 | 34.06 ± 3.45 | 33.74 ± 3.45 |

GSM8K is not reported for Llama 3-8B. Loop configuration per model is the one selected on WinoGrande; its exact values beyond Gemma 2-2B (layers 10-13) are not stated.

### Differences vs baseline (computed from Table I, in pp; my arithmetic, not in the paper)
| Benchmark | Model | Uniform | Moving Average | Auto-Align | Noise |
|---|---|---|---|---|---|
| WinoGrande | Gemma 2B | +0.78 | +0.31 | +0.23 | -0.24 |
| WinoGrande | Gemma 9B | +0.08 | 0.00 | 0.00 | 0.00 |
| WinoGrande | Llama 8B | +0.48 | +0.24 | -0.07 | -0.23 |
| Arc E | Gemma 2B | -0.29 | -0.12 | -0.08 | -0.46 |
| Arc E | Gemma 9B | 0.00 | +0.04 | 0.00 | 0.00 |
| Arc E | Llama 8B | -0.38 | -0.29 | -0.08 | -0.12 |
| Arc C | Gemma 2B | +0.59 | +0.08 | +0.17 | -0.17 |
| Arc C | Gemma 9B | +0.34 | +0.34 | -0.17 | -0.26 |
| Arc C | Llama 8B | -0.09 | 0.00 | +0.34 | -0.09 |
| GSM8K | Gemma 2B | +0.98 | +1.36 | +1.06 | -0.23 |
| GSM8K | Gemma 9B | +0.06 | -0.07 | -0.15 | -0.13 |
| HellaSwag | Gemma 2B | +0.19 | +0.13 | +0.09 | -0.01 |
| HellaSwag | Gemma 9B | +0.12 | +0.01 | +0.01 | +0.01 |
| HellaSwag | Llama 8B | +0.16 | +0.17 | +0.17 | 0.00 |
| MMLU | Gemma 2B | +0.56 | +0.42 | +0.20 | +0.04 |
| MMLU | Gemma 9B | +0.25 | +0.06 | +0.01 | +0.03 |
| MMLU | Llama 8B | -0.55 | -0.51 | -0.44 | -0.76 |

- Uniform: range -0.55 to +0.98 pp over 17 cells; 12 positive, 1 tied, 4 negative (count from the table above).
- Every difference is smaller than about 1.2 times the reported ± of its cell; the largest is Gemma 2B GSM8K under Moving Average, +1.36 with ± 1.21 (my arithmetic). The paper reports no significance test.

### Negative results (where looping hurts)
- Naive looping, Gemma2-2B, WinoGrande, loop count 3, all start-end pairs: every looped configuration is below baseline, some near chance. Numbers and heatmap are only in the repository, not in the paper.
- Regularized looping below baseline: Gemma 2B on ARC Easy (all three variants); Llama 8B on ARC Easy (all three variants) and MMLU (all three, -0.44 to -0.55 pp); Llama 8B ARC Challenge under Uniform (-0.09) and WinoGrande under Auto-Align (-0.07); Gemma 9B GSM8K under Moving Average and Auto-Align; Gemma 9B ARC Challenge under Auto-Align.
- The authors summarise Llama 3-8B (pre-norm) as responding less reliably than the Gemma models.
- Noise control: at or below baseline in 14 of 17 cells (count from the table above; the 3 positive cells are +0.01 to +0.04), which the authors read as evidence that gains are not due to perturbation magnitude alone.

### Other analyses
- Figure 1: heatmap of accuracy difference vs baseline (0.6875) over start-end pairs for Gemma 2-2B under uniform interpolation.
- Figure 2: 2D PCA of hidden-state trajectories for one randomly selected sample, with and without looping; the looped trajectory deviates slightly in the looped segment and the shift propagates to the output.

### Compute overhead
Not stated. (inference: one loop adds $(R-1)N$ block applications, from $K = L + (R-1)N$.)

### KV caching / generation
Not stated. GSM8K is evaluated generatively but cache handling under looping is not described.

### Medical or domain-specific benchmarks
None. MMLU is described as covering medicine among its 57 tasks, but no per-subject results are reported.

## What it establishes
- On Gemma2-2B / WinoGrande, naive block reapplication (loop count 3) lowers accuracy for every interval tested.
- Interpolating looped states with cached earlier states keeps accuracy close to baseline, within 1.4 pp in either direction, on all 17 reported cells (range -0.55 to +1.36 pp over the three variants, my arithmetic).
- With an interval chosen once on WinoGrande, the uniform rule gives small positive point differences on most Gemma cells (up to +0.98 pp) and mixed results on Llama 3-8B.
- A matched-magnitude noise control does not reproduce the positive differences.

## What it does NOT establish
- That the gains are statistically reliable: no test is reported and all differences are within about 1.2 times the per-cell ± (inference).
- An independent WinoGrande gain: the interval was selected on WinoGrande (inference).
- That the model's normalization scheme is the cause of the Gemma vs Llama difference: the two families differ in many ways and only three models are tested (inference).
- Anything about medical tasks, MedQA, Phi-family models or domain fine-tunes.
- Optimal $R$, $\eta$, or any per-task block choice: these are not swept or not reported.
- Behaviour during long generation with a KV cache, or the compute cost.
- A mechanism: the PCA figure is a single sample.

## Limitations stated by the authors
There is no limitations section. Caveats that appear in the text:
- Gains are described as modest.
- Llama 3-8B responds less reliably, attributed to architectural differences (pre-norm).
- Gemma-2B on ARC Easy is acknowledged as an exception to the improvements.
- Empirical behaviour is expected to depend on architecture details such as normalization placement.

## Implementation notes for med_wsp
- Code: github.com/jonathanlys01/looped-transformer (stated in the paper; not inspected here). Evaluation used lighteval.
- The rule is cheap to implement as a wrapped forward: keep a list of loop states, re-run blocks $s..e-1$ on the interpolated state, average. Memory: the cache holds up to $R+1$ hidden-state tensors of shape $T \times d$ for uniform and auto-alignment; moving average needs only $h^{(0)}$ and the current state (inference).
- Resolve the open choices before running (see "Things the text leaves open"): definition of $h^{(0)}$, which state is passed to block $e$, $R$, $\eta$. Record our choices in the experiment note, since results may depend on them (inference).
- Phi-3.5-mini is a pre-norm decoder, the same class as Llama 3-8B, the model that responded least reliably here (inference: a reason to expect small or mixed effects, not a prediction).
- Compute: models of 2B to 9B were used; hardware is not stated beyond an HPC allocation. Phi-3.5-mini (3.8B) with a 3 to 6 block loop fits a 36 GB M4 Max (inference).
- Scoring by length-normalized option likelihood is one forward pass per option without generation, so no KV-cache handling is needed for that protocol (inference).
- Gotcha: selecting the interval on the same questions used for reporting inflates the result; keep a separate selection split.

## Relevance to med_wsp (provisional)
- Second, independent source for the "Training-Free Reasoning" experiment, with a different stabiliser (state averaging instead of damped steps); see [[Layer looping]] for the comparison with [[Chen et al. 2026 - Training-Free Looped Transformers]].
- Effect sizes here (mostly under +1 pp, mixed sign on a pre-norm model) suggest that a MedQA study on Phi-3.5-mini and [[Corbeil et al. 2025 - MediPhi]] needs paired analysis and enough questions to resolve differences below 1 pp (provisional).
- The full start-end sweep (Figure 1) is a template for the per-model heatmaps we would need to ask whether a base model and its medical fine-tune prefer different blocks (provisional idea; the paper does not test this).
- The noise control is a useful baseline to copy: it separates "looping helps" from "any small perturbation of mid-layer states changes a few answers" (provisional).
- An agent such as GPT-6 Astra proposing blocks could be compared against this paper's exhaustive interval sweep on a small selection split (provisional idea).

## Open questions
- What exactly is $h^{(0)}$, and which state continues to block $e$?
- Which $R$ and $\eta$ produced Table I?
- What do the ± values denote (standard error, confidence half-width)?
- What intervals were selected for Gemma 2-9B and Llama 3-8B?
- Why is the Llama 3-8B MMLU baseline as low as 34.50?
- Does the uniform rule degrade for larger $R$, as Chen et al. report for their rendering of it?

## Ledger card
**Full citation:** Jonathan Lys, Vincent Gripon, Bastien Pasdeloup, Axel Marmoret, Lukas Mauch, Fabien Cardinaux, Ghouthi Boukli Hacene. "Inner Loop Inference for Pretrained Transformers: Unlocking Latent Capabilities Without Training." arXiv:2602.14759v2 [cs.LG], 2 Mar 2026. (IMT Atlantique, Lab-STICC; Sony Europe Ltd., Stuttgart Technology Center.)
**Source / DOI / URL:** arXiv:2602.14759; local text `papers/pdf/text/arxiv_2602.14759.txt`, PDF `papers/pdf/arxiv_2602.14759.pdf`.
**Why I read it:** Second method for looping layers of a frozen model; need the exact inner-loop rule, block selection and effect sizes.
**Problem addressed:** Whether extra refinement steps can be added at inference to a frozen pretrained transformer by re-applying a block range.
**Method / representation:** Re-apply blocks $s..e-1$ $R$ times; interpolate looped states with cached states, $\hat h^{(t)} = \sum_i \alpha_i h^{(i)}$ (uniform, moving average with $\eta$, softmax auto-alignment); naive looping as a special case.
**Assumptions:** Depth acts as iterative refinement of a shared latent state; looped states must stay near the standard-pass activation distribution.
**Evidence used by the paper:** experiments: Table I over 6 benchmarks and 3 models, a naive-looping sweep, a noise control / analysis: start-end heatmap for Gemma 2-2B, one PCA trajectory plot / datasets or systems: Gemma 2-2B, Gemma 2-9B, Llama 3-8B; WinoGrande, ARC Easy, ARC Challenge, GSM8K, HellaSwag, MMLU.
**Safe citation claims:** "Lys et al. (2026) report that naive inner looping lowers accuracy and that interpolating looped states with cached states yields modest gains on Gemma 2 models"; "they select the loop interval by a sweep on WinoGrande and find it at roughly 40-60% of depth"; "they report less reliable results on the pre-norm Llama 3-8B".
**Unsafe / unsupported citation uses:** "inner looping significantly improves reasoning"; "looping improves medical question answering"; "uniform interpolation is the best stabiliser in general"; any specific claim about $R$ or $\eta$ values.
**Relation to our project:** closest prior art | overlapping method
**Verification status:** full text read: yes; key claim checked: yes for equations 5-12 and Table I against the PDF; repository not inspected; results not reproduced.

## Links
- [[Layer looping]]
- [[Chen et al. 2026 - Training-Free Looped Transformers]]
- [[Corbeil et al. 2025 - MediPhi]]
- [[Activation patching]] (hook-based interventions on the residual stream, related tooling)
