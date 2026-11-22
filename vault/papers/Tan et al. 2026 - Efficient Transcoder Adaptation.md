---
type: paper
title: "Efficient Transcoder Adaptation for Fine-Tuned Models: Revealing Medical Reasoning Mechanisms in Large Language Models"
short: "Tan et al., 2026"
authors: [Zhouxing Tan, Hanlin Xue, Yulong Wan, Ruochong Xiong, Xu Chu, Xiang Li, Junfei Liu]
year: 2026
venue: "AAAI-26, Proceedings of the AAAI Conference on Artificial Intelligence 40(39), pp. 33196-33204"
url: "https://ojs.aaai.org/index.php/AAAI/article/view/40604"
local_text: "papers/pdf/text/aaai2026_transcoder_adaptation.txt"
tags: [paper, transcoders, fine-tuning, medical, attribution-graphs, gemma-2-2b, closest-prior-art]
read_status: full-text-read
---

# Tan et al., 2026 - Efficient Transcoder Adaptation (FTFA + PAT)

> Reading conventions: plain statements paraphrase or quote the paper. Anything marked "(inference)" is mine. "Not stated" means I looked for it in the 9-page AAAI PDF and did not find it. The paper defers several details to an "extended version" that is not in the vault. Numbers marked "read from plot" are my approximate readings of scatter plots, not values printed by the authors.

## TL;DR
The paper proposes two coupled procedures for getting [[Transcoders]] onto a fine-tuned model cheaply. FTFA ("Medical Fine-Tuning with Frozen Attention Layers") fine-tunes only the MLP/FFN parameters of Gemma2-2b on medical data with all attention frozen. PAT ("Posterior Adaptation Transcoders") then takes the existing per-layer transcoders of the base model and adapts them: stage 1 fine-tunes them on the same data with the same fine-tuning loss, stage 2 refines them with the ordinary transcoder loss on a sampled subset. Stated cost: 1M tokens and 0.2 H100 hours, against "approximately 1B training tokens and 210 H100 hours" for training from scratch ("over 1000x"), at 16,384 features per layer over 26 layers. FTFA reaches 52.27 accuracy / 56.36 BERTScore versus 50.79 / 51.21 for base Gemma-2-2b and 50.06 / 54.18 for full fine-tuning (SFT). Attribution-graph case studies are read as showing two parallel mechanisms (pattern matching and multi-hop reasoning), with the fine-tuned model using more correct multi-hop paths.

## Problem
In the paper's framing: transcoder-based circuit analysis works on base models, but "training transcoders from scratch for each fine-tuned model is computationally expensive, while directly applying base model transcoders to fine-tuned models may not capture the nuanced changes introduced by domain adaptation". They state that transcoder training requires "compute resources comparable to pretraining". The goal is a low-cost way to obtain transcoders for a fine-tuned model, applied to medical diagnosis because verbalized reasoning can differ from internal computation.

## Method (step by step)

Notation: $M_1$ = base Gemma2-2b; $M_2$ = medical fine-tune of $M_1$; $\mathrm{MLP}_1, \mathrm{MLP}_2$ = their MLP modules (per layer); $TC_1, TC_2$ = the corresponding transcoders; $D$ = the fine-tuning dataset (Table 1 below).

1. **Transcoder objective (Eq. 1).** One transcoder is trained per MLP layer to reproduce the MLP output from the MLP input:
   $L_{TC} = \lVert \mathrm{MLP}(x) - \mathrm{TC}(x) \rVert_2^2 + \lambda_1 \lVert \theta_{TC}(x) \rVert_1$
   where $\mathrm{TC}(x)$ is the transcoder output for input $x$ and $\theta_{TC}(x)$ are the sparse activation coefficients. Sizes: "each hidden layer has a size of 2K" for Gemma2-2B and the transcoder expands to "16K dimensions" (16,384 features per layer, 26 layers). The functional form used in the theory section is $f(x) = W_1\,\mathrm{ReLU}(W_2 x + b_2) + b_1$. Value of $\lambda_1$: not stated.
2. **FTFA: fine-tune MLPs only (Eq. 2).** Freeze "all attention components within Transformer layers and update only the MLP parameters":
   $\mathcal{L}(\theta_{MLP}) = \mathbb{E}_{(x,y)\sim D}\left[\ell\left(f(x;\theta^{0}_{Attn},\theta_{MLP}),\,y\right)\right]$
   with $\theta^{0}_{Attn}$ the frozen attention parameters, $\theta_{MLP}$ trainable, $\ell$ "a standard supervised learning objective, such as cross-entropy", and $D = \{(x_i,y_i)\}_{i=1}^{N}$ user inputs and targets. Whether embeddings, unembedding and normalization parameters are frozen: not stated. Optimizer, learning rate, batch size, epochs, prompt format: not stated.
3. **Starting point for PAT.** $TC_1$ = pre-trained per-layer transcoders of the base model. Figure 1 names them as "Neuronpedia per-layer transcoders" for Gemma2-2b; the text cites Lieberum et al. 2024 (Gemma Scope) and the circuit-tracer library (Hanna et al. 2025).
4. **PAT stage 1: fine-tune the base transcoders with the task loss on the same data.** The "key observation" is that $TC_2$ can be approximated by $TC_2'$ with parameters
   $\theta' = \arg\min_{\theta_{TC}} \mathbb{E}_{x\sim D}\left[L\left(f_{\theta_{TC}}(x),\,y\right)\right], \quad \theta_{TC}^{(0)} = \theta_{TC_1}$
   i.e. initialize at $TC_1$ and continue training on $D$ with the fine-tuning loss $L$ (Figure 3 labels this arrow "L: fine-tuning loss, D: medical data, Stage 1" and its result "$TC_2' \approx \mathrm{MLP}_2$"). (inference) Operationally this means the transcoders sit in the model in place of the MLPs, attention stays frozen, and the transcoder weights receive the same supervised gradient that the MLPs received during FTFA. The paper does not say whether all 26 transcoders are trained jointly, whether the sparsity term is kept in stage 1, or whether error terms are added.
5. **Theoretical justification (Eq. 3, Eq. 4).** Claim to prove: $\lim_{m\to\infty} \mathrm{MLP}_2(x) = \mathrm{TC}_2'(x)$ almost everywhere, with $m = \min\{m_{MLP_2}, m_{TC_2'}\}$ the hidden dimension. Argument: under Neural Tangent Kernel theory the training trajectory of a wide network is
   $f_\theta^{(t)}(x) = f_\theta^{(0)}(x) - \eta \sum_{s=0}^{t-1} K_\theta(x,X)\left(f_\theta^{(s)}(X) - Y\right)$, with $K_\theta(x,x') := \nabla_\theta f(x)^{T}\nabla_\theta f(x')$,
   and as $m\to\infty$ the kernel converges to a deterministic $K^{\infty}(x,x')$ that is the same for networks with identical architecture and activation. MLP and transcoder are said to share the form $W_1\,\mathrm{ReLU}(W_2x+b_2)+b_1$, their initial outputs are "almost everywhere equal" (because $TC_1$ was trained to match $\mathrm{MLP}_1$), so the same data and loss give the same trajectory. This is why attention must be frozen: both networks must see the same inputs and the same loss.
6. **PAT stage 2: refine with the ordinary transcoder loss.** Because real networks are finite width and the $\lVert\theta_{TC}\rVert_1$ constraint adds "optimization difficulty", $TC_2'$ is further trained "on $D_{sample}$, based on the original transcoder training loss Equation (1)", where $D_{sample}$ is "a sampled subset from the larger dataset required for transcoder training". Figure 3 draws stage 2 as $TC_2'$ feeding the transcoder loss $L_{TC_2}$ against $\mathrm{MLP}_2$, producing $TC_2$. Size and source of $D_{sample}$: not stated beyond the overall budget (1M tokens total; split between stages not stated).
7. **Feature re-interpretation.** Because adapted features may have changed meaning: Direct Logit Attribution of each feature through the unembedding, plus inspection of activation patterns "across diverse textual inputs from multiple datasets"; semantically similar features are merged into super-nodes. The pipeline "maintains consistency with Neuronpedia's methodology"; details are in the extended version.
8. **Attribution graphs.** Replace the MLPs of base and fine-tuned model with their per-layer transcoders (base: $TC_1$; fine-tuned: PAT), build feature-level [[Attribution graphs]], prune, merge nodes, compare. The attribution algorithm itself is not specified here; it follows [[Ameisen et al. 2025 - Circuit Tracing]].
9. **Auxiliary analyses.** (a) PCA shift: PCA with $n=2$ on hidden states from each layer of $M_1$ and $M_2$ gives layer-wise coordinates $(m_{i,1}, m_{i,2})$; PC1 shift $\Delta m_{i,1} = m_{i,1}^{(2)} - m_{i,1}^{(1)}$; global shift $d = \lVert z^{(2)} - z^{(1)} \rVert_2$ with $z^{(*)}$ the mean PCA coordinates across layers. (b) Feature categories: input-related, correct reasoning, wrong reasoning, non-medical. Input-related and non-medical are assigned by "a trained classifier on 1,000 medical diagnostic queries"; correct and wrong reasoning are "manually annotated via attribution graph analysis on a curated set of 10 representative samples".

Ambiguity in the source: the FTFA section also says "we initialize each transcoders $TC_1$ with the parameters of its corresponding fine-tuned MLP module $MLP_1$". This conflicts with the PAT section, where $TC_1$ is the base model's pre-trained transcoder and subscript 1 is the base model, and a 16K-wide transcoder cannot literally copy a differently sized MLP. I read the PAT section and Figure 3 as authoritative (inference).

## Experiments and numbers

**Model.** Gemma2-2b ("Gemma-2-2b" in tables), 26 layers, transcoder width 16,384 per layer.

**Fine-tuning data (Table 1, "medical datasets used for FTFA").**

| Dataset | Count | Task type |
|---|---|---|
| MEDQA | 10178 | Multiple-Choice |
| Symptom2Disease | 853 | Question Answering |
| DDXPlus | 13310 | Question Answering |

Total 24,341 examples (my sum). Note that MedQA is training data here, not the test set.

**Q1 - downstream performance (Table 2).** Benchmark: "a medical benchmark (Perets et al. 2025)"; the reference entry is CUPCase. Test-set size, seeds and confidence intervals: not stated.

| Model | Accuracy | BERTScore |
|---|---|---|
| Gemma-2-2b-FTFA | 52.27 (+1.48) | 56.36 (+5.15) |
| Gemma-2-2b-SFT (full-parameter fine-tuning) | 50.06 (-0.73) | 54.18 (+2.97) |
| Gemma-2-2b | 50.79 | 51.21 |
| Llama-3.2-3B | 60.5 | 60.19 |
| Llama-3.1-8B | 65.55 | 61.36 |
| Qwen2.5-7B-Instruct | 68.72 | 63.51 |
| II-Medical-8B | 66.45 | 60.26 |
| MedFound-7B | 63.29 | 60.15 |

**Q2 - representation change.** PCA shift $d$ per disease as printed in Figure 4: HIV 3.19, Diabetes 3.80, Arthritis 4.14, Cancer 4.29, Lymphoma 4.13, Migraine 5.66, Psoriasis 3.66, Tuberculosis 3.08. The text says diabetes and cancer "show relatively small shifts" and HIV, Migraine and Psoriasis "exhibit larger shifts"; the printed values only support this for Migraine (HIV and Psoriasis are below Diabetes and Cancer). Also the caption defines $d$ as "the average absolute shift along PC1" while the text defines it as the L2 norm above. Feature proportions (Figure 5a): medical-feature activation is U-shaped over depth, strong in layers 1-8 and 20-25, reduced in 10-15; FTFA raises it "across the board, with notable gains in Layers 7-9, 16, and 19-23". Figures 5b/5c: the FTFA model activates more input-related and correct-reasoning features and fewer non-medical and wrong-reasoning features. No numeric table is given.

**Q3 - PAT accuracy and efficiency (Figures 6 and 7; values read from plot, approximate).** "Base-Base" points are per-layer transcoders on an 18-layer model taken from Ameisen et al. 2025; "FT-Base" is fine-tuned Gemma2-2b with unmodified base transcoders; "Ours" is FTFA Gemma2-2b with PAT. Metric definitions are not given in the paper; (inference) they are replacement-model next-token disagreement and KL to the real model's output, as in [[Ameisen et al. 2025 - Circuit Tracing]].

| Configuration | L0 | Top-1 error rate | KL divergence |
|---|---|---|---|
| Base-Base-16k-18L | ~160 | ~0.90 | ~4.0 |
| Base-Base-65k-18L | ~180 | ~0.78 | ~2.6 |
| Base-Base-130k-18L | ~160 | ~0.75 | ~2.3 |
| Base-Base-524k-18L | ~150 | ~0.72 | ~2.25 |
| FT-Base-16k-26L | ~264 | ~0.995 | ~7.9 |
| Ours-16k-26L (PAT) | ~169 | ~0.885 | ~3.7 |

Stated in text: base transcoders on the fine-tuned model give "nearly zero Top-1 accuracy and poor KL divergence"; PAT is "comparable or even superior" to from-scratch training. Cost: "traditional transcoders training requires approximately 1B training tokens and 210 H100 hours, our approach accomplishes similar results with only 1M tokens and 0.2 H100 hours" (1000x in tokens, 1050x in hours by my arithmetic).

**Q4 - case study (Figure 8).** Prompt tokens fever, weight loss, thrush. Base model: pattern-matching edges from symptom features to HIV exist, but a multi-hop chain fever -> flu -> pneumococcal infection -> tuberculosis gets more weight and the output is tuberculosis (wrong). FTFA model: a "Viruses and Pathogens" supernode -> Opportunistic Infections -> Immune Deficiency -> HIV, output HIV (correct). Figure nodes carry (layer, feature) ids, e.g. "pneumococcal vaccine (23, 12195)" appears in both graphs. Figure 1 is a base-model diabetes example that is correct via pattern matching.

## What it establishes
- On Gemma-2-2b, MLP-only fine-tuning on about 24k medical examples did not hurt and slightly improved the paper's benchmark scores relative to base and to full fine-tuning (single numbers, no variance).
- Unmodified base-model transcoders, used as a full replacement model for a fine-tuned model, fail badly on the paper's metrics (Top-1 error near 1.0).
- After PAT, 16k-wide transcoders for the fine-tuned model reach an (L0, Top-1 error, KL) point close to the 16k per-layer baseline that Ameisen et al. report for a different 18-layer model, at the stated cost of 1M tokens and 0.2 H100 hours.
- Qualitatively, attribution graphs for one prompt where base is wrong and the fine-tune is right show different intermediate supernodes in the two models.

## What it does NOT establish
- That PAT works when attention weights changed. The method and its justification assume an attention-frozen, MLP-only fine-tune that the authors made themselves. LoRA, merged, or third-party fine-tunes are not tested.
- That PAT matches from-scratch transcoders on the same model. No from-scratch transcoder is trained on the FTFA model; all "from scratch" points come from an 18-layer model in another paper, and no point is shown for base Gemma2-2b with its own base transcoders.
- That stage 1 is necessary. There is no ablation of stage 1 only, stage 2 only, or warm-start plus Eq. 1 alone.
- That the adapted replacement model is faithful in absolute terms. (inference) A Top-1 error near 0.885 means the replacement model disagrees with the real model on most tokens, like the 16k baseline it is compared to.
- Causal validity of the graphs. No ablation, steering or [[Activation patching]] experiment is reported in the main paper.
- Generality of the "two mechanisms" claim. It rests on case studies and 10 manually annotated samples.
- Anything about attention heads. They are frozen by design and not analysed.
- Anything about weight edits, transferring a computation into the base model, or LLM-agent-driven circuit discovery. None of these appear.
- Any systematic study of items the fine-tune gets right and the base gets wrong. There is one such case in the main text.
- Reconstruction MSE or explained variance per layer, dead-feature rates, or interpretability scores: not reported.
- Reproducible hyperparameters: none are given in the paper (the code link is the only route).

## Limitations stated by the authors
- Case study dependency: "our mechanistic insights rely primarily on case studies", which "limits the scalability of our analysis method".
- Model scale and generalizability: not validated beyond Gemma2-2b or on other architectures.
- Theoretical constraints: the NTK argument assumes infinite width; real networks are finite width with sparsity constraints "that may deviate from theoretical predictions".

## Implementation notes for med_wsp
- Code: https://github.com/ColinNeverLand/llm-medical-reasoning-mechanisms (stated in the paper; not inspected for this note). Base transcoders came from Neuronpedia / Gemma Scope; graphs via circuit-tracer.
- PAT needs three things: (1) pre-trained transcoders for the base model, (2) a fine-tune whose attention weights equal the base model's, (3) the fine-tuning data and loss.
- Status of these for our pair (inference): (1) neither source read for this note mentions transcoders for Phi-3.5-mini, so this must be checked, and if none exist the expensive base training is still required. (2) and (3): the local MediPhi model card (`papers/pdf/mediphi_model_card.md`, see [[Corbeil et al. 2025 - MediPhi]]) describes five fine-tuned experts merged with SLERP and BreadCrumbs; it does not say attention was frozen, so the first check is a per-module weight diff. If attention weights differ, stage 1 has no justification from this paper.
- (inference) Stage 2 alone (warm start from a base transcoder, then Eq. 1 against the fine-tuned MLP's inputs and outputs) does not need frozen attention or the fine-tuning data, only activations. The paper does not evaluate it in isolation.
- (inference) Size on our hardware, assuming Phi-3.5-mini has 32 layers and residual width 3072 (general knowledge, verify in config.json): one 16,384-feature transcoder is $2 \times 16{,}384 \times 3{,}072 \approx 100.7$M parameters, about 0.4 GB in fp32 and about 1.2 GB with Adam state, so per-layer training fits easily. All 32 layers are about 3.2B parameters, 12.9 GB in fp32 and about 38.7 GB with Adam state, which does not fit in 36 GB next to the model; joint stage 1 training would need a subset of layers, narrower dictionaries or a lighter optimizer. Cached MLP input and output for one layer cost about 12.3 KB per token in fp16, so 1M tokens is about 12.3 GB per layer on disk.
- Cost caveat (cross-check outside this paper): the local text of Ameisen et al. (`papers/pdf/text/circuit_tracing_methods_2025.txt`) gives "2M features and 1B train tokens ... roughly 210 H100 hours at 50% flops efficiency" as an estimated cost of a cross-layer transcoder on Gemma 2 2B. (inference) Tan et al.'s from-scratch reference appears to be this estimate, not a measured cost for 26 per-layer transcoders of 16,384 features (425,984 features in total), so the 1000x ratio is indicative only. Tan et al. give no citation in that sentence.

## Relevance to med_wsp (provisional)
- Closest prior art on the question "what does a medical fine-tune change inside the model", answered with transcoders and attribution graphs on a base versus fine-tune pair.
- Facts that separate it from our plan: different model (Gemma2-2b, not Phi-3.5-mini); the fine-tune is the authors' own and constrained to MLP-only updates; MedQA is in their training set; comparison is by case study; no weight-difference analysis (the PCA analysis is on hidden states); no patching; no crosscoder; no agent; no weight edit or transfer.
- Provisional idea: if the Phi-3.5 versus MediPhi weight diff is concentrated in MLPs for some layers, a PAT-like warm start is a candidate for those layers only. If not, [[Crosscoders]] ([[Lindsey et al. 2024 - Sparse Crosscoders]]) do not need frozen attention.
- Provisional idea: their finding that base transcoders break on the fine-tune is a warning for any Astra pipeline that reuses base-model dictionaries on MediPhi without checking replacement metrics first.
- Provisional idea: their HIV case (base wrong, fine-tune right) is the unit of analysis we want to scale up over the MedQA failure set.

## Open questions
- Does PAT keep feature identity? Figure 8 reuses ids such as (23, 12195) in both graphs, which suggests index alignment (inference), yet the paper adds a re-interpretation step.
- How much of the PAT result is stage 1 versus stage 2?
- What are the PAT replacement metrics for base Gemma2-2b with its own base transcoders on the same prompts? Without it the PAT point has no same-model reference.
- Which "fine-tuned" model is FT-Base-16k-26L: FTFA or SFT? Not stated.
- Do pre-trained transcoders exist for Phi-3.5-mini, and how different are MediPhi's attention weights from the base?
- How large is the gap between a merged fine-tune like MediPhi and the single-dataset supervised setting assumed by Eq. 2?

## Ledger card
**Full citation:** Tan, Z.; Xue, H.; Wan, Y.; Xiong, R.; Chu, X.; Li, X.; Liu, J. 2026. Efficient Transcoder Adaptation for Fine-Tuned Models: Revealing Medical Reasoning Mechanisms in Large Language Models. Proceedings of the AAAI Conference on Artificial Intelligence 40(39): 33196-33204.
**Source / DOI / URL:** https://ojs.aaai.org/index.php/AAAI/article/view/40604 ; DOI 10.1609/aaai.v40i39.40604 (from the local landing page file); PDF `papers/pdf/aaai2026_transcoder_adaptation.pdf`.
**Why I read it:** closest prior art for interpreting a medical fine-tune against its base with transcoders.
**Problem addressed:** transcoders for a fine-tuned model are costly to train from scratch and base transcoders do not transfer.
**Method / representation:** MLP-only fine-tuning with frozen attention (FTFA); two-stage adaptation of base per-layer transcoders (PAT); attribution graphs with supernodes.
**Assumptions:** attention weights unchanged by fine-tuning; base transcoders exist and approximate the base MLPs; fine-tuning data available; NTK infinite-width regime as justification.
**Evidence used by the paper:** experiments: Table 2 accuracy/BERTScore, Figures 6-7 replacement metrics, PCA shift, feature-category proportions / analysis: NTK sketch, attribution-graph case studies / datasets or systems: Gemma2-2b, MedQA, Symptom2Disease, DDXPlus for fine-tuning, CUPCase benchmark, Neuronpedia transcoders.
**Safe citation claims:** "Tan et al. adapt base-model per-layer transcoders to an attention-frozen medical fine-tune of Gemma2-2b and report 1M tokens and 0.2 H100 hours for the adaptation"; "They report that unadapted base transcoders give near-zero top-1 accuracy as a replacement model for the fine-tuned model"; "They describe pattern-matching and multi-hop pathways in attribution-graph case studies".
**Unsafe / unsupported citation uses:** "PAT works for arbitrary fine-tunes"; "PAT matches from-scratch transcoders on the same model"; "the 1000x saving is a measured like-for-like comparison"; "fine-tuning causally creates multi-hop medical reasoning"; any claim about MedQA test accuracy or about Phi-family models.
**Relation to our project:** closest prior art
**Verification status:** full text read: yes; key claim checked: partly (numbers checked against the PDF and figures; the code repository and extended version were not checked)

## Links
- Concepts: [[Transcoders]], [[Attribution graphs]], [[Crosscoders]], [[Activation patching]]
- Papers: [[Ameisen et al. 2025 - Circuit Tracing]] (source of the 18-layer baselines and the graph method), [[Lindsey et al. 2024 - Sparse Crosscoders]] (alternative base versus fine-tune comparison), [[Corbeil et al. 2025 - MediPhi]] (our fine-tuned model)
- Project side not covered by this paper: [[Weight-space programming]], [[Takehi et al. 2026 - Weight-Space Programming]], [[Li et al. 2026 - WeightCode]]
