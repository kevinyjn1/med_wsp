---
type: paper
title: "A Modular Approach for Clinical SLMs Driven by Synthetic Data with Pre-Instruction Tuning, Model Merging, and Clinical-Tasks Alignment"
short: "Corbeil et al., 2025"
authors: [Jean-Philippe Corbeil, Amin Dada, Jean-Michel Attendu, Asma Ben Abacha, Alessandro Sordoni, Lucas Caccia, François Beaulieu, Thomas Lin, Jens Kleesiek, Paul Vozila]
year: 2025
venue: "ACL 2025 (Volume 1: Long Papers), pp. 19352-19374; text read = arXiv:2505.10717v2"
url: "https://arxiv.org/abs/2505.10717"
local_text: "papers/pdf/text/arxiv_2505.10717.txt"
local_card: "papers/pdf/mediphi_model_card.md"
tags: [paper, mediphi, phi-3-5, model-merging, medical-llm, medqa, weight-diff]
read_status: full-text-read
---

# Corbeil et al., 2025 - MediPhi

> Sources read: (1) the full extracted text of arXiv:2505.10717v2 including all appendices (23 pages; tables, equations and figures cross-checked against the PDF render of pages 1-9, 13-15, 21-23), (2) the Hugging Face model card of `microsoft/MediPhi`.
> Conventions in this note: unmarked statements are what the sources say. "(inference)" marks my own reasoning. "[UNVERIFIED-BACKGROUND]" marks general knowledge that is NOT in the two supplied sources and has not been checked. "not stated" means neither source gives it.

## TL;DR
- The paper adapts `Phi-3.5-mini-instruct` (3.8B parameters) to medical and clinical NLP with a modular pipeline: (1a) five domain experts obtained by continual pre-training (pre-instruction tuning, PIT; plain DAPT for PubMed), (1b) each expert merged back into the base with SLERP, then all five combined into one model, "MediPhi", with BreadCrumbs, (2) alignment of MediPhi on the synthetic MediFlow instruction dataset with SFT then DPO, giving "MediPhi-Instruct".
- Headline, CLUE+ average over 12 clinical datasets: Phi-3.5 36.5 -> MediPhi 39.3 -> MediPhi-SFT 43.0 -> MediPhi-Instruct 43.4 (+6.9 absolute, 18.9% relative).
- The Hugging Face repo `microsoft/MediPhi` is the BreadCrumbs merge (end of stage 1b). It is NOT the MediFlow-aligned model.
- MedQA appears only in appendix Table 13, with no protocol: Phi-3.5 0.486, MediPhi 0.491, MediPhi-SFT 0.536, MediPhi-Instruct 0.548.
- Neither source says whether fine-tuning was full-parameter or adapter-based, and neither gives the merge hyperparameters (only that SLERP proportions were picked from 10%, 25%, 50%).

## Problem
In the paper's framing: large models such as GPT-4 are too costly and slow for clinical deployment; small language models (SLMs) are a cheaper alternative but need biomedical domain adaptation, which is hard at small capacity; clinical data is sensitive and tightly licensed; and existing medical LLMs do well on multiple-choice exams but show a gap on real clinical tasks (the paper cites the CLUE benchmark results of Dada et al. 2024 and others for this). The paper proposes a modular training framework for clinical SLMs and a synthetic instruction dataset (MediFlow) to address data unavailability.

## Method (step by step)
Short overview; the exact stage-by-stage training and merge details are in the next section.

1. Start from `Phi-3.5-mini-instruct`.
2. Continual pre-training into five experts from five corpus groups (PubMed, Clinical, MedCode, Guidelines, MedWiki). Three knowledge-acquisition methods are compared: DAPT (next-token prediction on raw domain text), Explainer (fine-tune on GPT-4o generated textbook-like material), PIT (Jiang et al. 2024: first fine-tune on generated task data, then on task data concatenated with the source documents). PIT task types tried: QA, summarization, NER, relation extraction; summarization is used for the rest of the paper. A footnote says PIT involves no explicit instruction data or chat-style formatting.
3. Merge each expert with the base using SLERP (two-model merge) to recover abilities degraded by PIT (instruction following, long context, multilingual).
4. Merge the five experts into one model using a multi-model merge operator (Task Arithmetic, TIES, BreadCrumbs compared), with the merge configuration optimized by MergeKit's evolutionary algorithm on 12 synthetic multiple-choice validation sets (stops after 500 evaluations, objective = average accuracy). BreadCrumbs result is named MediPhi.
5. Generate MediFlow with GPT-4o: a meta-prompt conditioned on input-data type, task type, difficulty level, output format and temperature (1.0 for 70% of the data, 1.25 for 30%); 10 instructions per request with four input-output pairs each (appendix A.4 says 3 or 4 pairs). Result: 2.5M instruction triplets, 14 task types, 36 input-data types expanded to 98 fine-grained types, plain text or JSON output. Reported token statistics: instructions 301 +/- 295, inputs 76 +/- 67, outputs 79 +/- 66.
6. Score every instruction with GPT-4o mini as judge (chain-of-thought, $M = 5$ samples at temperature 1.0) on five criteria on a 1-4 scale. Score of criterion $j$ as printed: $S_j = \frac{1}{M}\sum_{i=1}^{M} c_i \cdot s_{ij}$, with $s_{ij} \in \{1,2,3,4\}$, counts $c_i \in [0, M]$, $\sum_{i=1}^{M} c_i = M$. A heuristic keeps the top-K by quality (the 800k subset).
7. Build MediFlow-DPO: filter to around 85k top-quality instructions stratified by task type, input-data type and output format; sample 3 pairs for the top 20k instructions, 2 pairs for the next 25k, 1 pair for the last 40k; after filtering 130,852 triplets. GPT-4o (temperature 1.0) writes a marginally wrong "rejected" output using one of 12 error types.
8. Align MediPhi with SFT on MediFlow (MediPhi-SFT), then DPO on MediFlow-DPO (MediPhi-Instruct).
9. Model selection metrics: CLUE+ average, #DG (number of the 12 datasets with a gain over base), and the coefficient of variation of gains/losses

$$CV\,\Delta = \frac{\sqrt{\mathbb{E}_{d\sim D}\left[(\delta_d - \mu_d)^2\right]}}{|\mu_d|}$$

where $\delta_d$ is expert score minus base score on dataset $d$ and $\mu_d = \mathbb{E}_{d\sim D}[\delta_d]$. Small values mean uniform gains or losses; large values mean changes concentrated on few datasets.

Merge validation sets (appendix A.3): 12 sets, GPT-4o generated multiple-choice questions on: clinical case, clinical knowledge, medication, ICD10CM code definitions, radiology report, clinical NLI, QA on discharge letter, medical codes of discharge letter, problem list from clinical notes, summarization of patient inquiry, QA on medical consultation, QA on multiple EHR documents. Diversity control: embed question plus context, cluster with HDBSCAN, keep outliers plus one sample per cluster, up to 1,200 samples per set.

## How MediPhi's weights were produced from Phi-3.5-mini-instruct

### Pipeline
1. **Base.** `microsoft/Phi-3.5-mini-instruct`, 3.8B parameters. Card: dense decoder-only Transformer, same tokenizer as Phi-3 Mini, default context length 128K tokens.

2. **Stage 1a - five domain experts (continual pre-training of the base, each expert trained independently from the base).**

   | Expert | Training data (paper Table 1; HF dataset ids from the card) | #Docs | #Tokens | Method |
   |---|---|---|---|---|
   | PubMed | PMC scientific articles (commercial-use segment) | 3.8M | 42B | DAPT (Sec. 3.1.2 says this expert follows standard DAPT because the corpus is orders of magnitude larger) |
   | | PMC Abstract, scientific abstracts (`ncbi/pubmed`) | 36M | 6B | |
   | Clinical | PMC-Patients v2, patient summaries (`zhengyun21/PMC-Patients`, commercial-friendly CC subset) | 156k | 130M | PIT |
   | | Asclepius, synthetic clinical documents (`starmpcc/Asclepius-Synthetic-Clinical-Notes`) | 80k | 44M | |
   | | NoteChat, doctor-patient conversations (`akemiH/NoteChat`) | 80k | 72M | |
   | | MTSamples, clinical documents | 5k | 4M | |
   | MedCode | ICD9CM, ICD9PROC, ICD10CM, ICD10PROC, ATC webpages | 206k | 257M | PIT |
   | Guideline | Medical guidelines websites (`epfl-llm/guidelines`) | 37k | 90M | PIT |
   | MedWiki | Medical Wikipedia (`jpcorb20/medical_wikipedia`) | 80k | 80M | PIT |

   Appendix A.1 gives the PubMed group as about 48B tokens.

   - **PIT recipe (Sec. 3.1.2).** GPT-4o generates task data for each document. Phase 1: fine-tune on the generated outputs of a single task for 1 epoch, cosine scheduler, peak LR 1e-4 (multi-element outputs such as several QA pairs are concatenated into one sequence separated by EOS tokens). Phase 2: 2 epochs, another cosine scheduler, peak LR 3e-4, on the concatenation of task data and the original documents with EOS separators. Task used after the ICD10CM case study: summaries. Appendix A.2: PIT experts are trained one document at a time; all experts avoid padding and cross-document attention. Other PIT hyperparameters (optimizer, batch size, warmup, max tokens): not stated (Table 6 explicitly excludes PIT).
   - **DAPT (PubMed expert, DataMix baseline).** Sec. 3.1.2: 1 epoch, cosine scheduler, max LR 5e-5. Appendix Table 6 (CPT hyperparameters, excluding PIT): max tokens 4,096; AdamW; linear warmup then cosine; 35 warmup steps; max LR 1e-4; 1 epoch; NEFTune alpha 5; batch size per GPU 16; 16 GPUs; gradient accumulation 2; effective batch size 512. Best-Fit Packing is used for the PubMed expert and the DataMix baseline. The 5e-5 (text) vs 1e-4 (Table 6) mismatch is not explained.
   - **Explainer (tried for MedCode).** 2 epochs, cosine scheduler, max LR 1e-4, on GPT-4o generated textbook-like explainers. Whether the final MedCode expert used explainers, PIT summaries, or both: not stated explicitly (the paper says the rest of the paper applies PIT with summaries unless otherwise stated).
   - **Full fine-tuning vs LoRA/DoRA/other adapters: not stated.** A keyword search of the full paper text for LoRA, DoRA, adapter, PEFT, rank, alpha, target modules, frozen/freeze, trainable found nothing relevant, and the card is equally silent. The paper only says "fine-tuning"/"trained". Ranks, alpha, target modules, which layers were trainable: not stated.
   - Library stack: Hugging Face transformers, trl, accelerate, datasets, multi-node.

3. **Stage 1b-i - SLERP of each expert with the base (Fig. 2a).** Merging proportions were chosen from 10%, 25% or 50% using the validation sets. The proportion chosen for each expert: not stated. The ICD10CM case study (Fig. 5) uses SLERP 50%. Whether the percentage is the expert's or the base's share, and whether it is uniform across layers/tensors: not stated. These five SLERP-merged models are the "PubMed, Clinical, MedWiki, Guideline, MedCode" experts of the card. Evidence that this step matters (Guideline ablation, CLUE+ average): with PIT and SLERP 39.2; without SLERP 27.2; without PIT 33.0; without both 25.2; base 36.5.

4. **Stage 1b-ii - multi-expert merge into MediPhi (Fig. 2b).** Operators compared: Task Arithmetic, TIES, BreadCrumbs. Configuration search: evolutionary algorithm via MergeKit, terminated after 500 evaluations, guided by average accuracy on the 12 synthetic validation sets. Selected: **BreadCrumbs** (CLUE+ 39.3, gains on 11 of 12 datasets, CV Delta 1.5) over Task Arithmetic (39.4, 9, 1.9) and TIES (39.3, 7, 1.7). BreadCrumbs hyperparameters (per-expert weights, density, outlier fraction, any per-layer settings, normalization): **not stated**. DARE appears only in related work and in the BioMistral-7B-DARE baseline; it is not used for MediPhi.
   - Ambiguity: the Fig. 1 caption says experts are merged first with the base and then together, which implies the SLERP-merged experts are the inputs of the BreadCrumbs merge; Fig. 2b draws experts coming straight out of domain-specific pre-training into the merging operator together with the base. Which exactly was used: not stated unambiguously. (inference: the SLERP-merged experts, following the Fig. 1 caption and the order of the card's summary.)

5. **Stage 2a - SFT on MediFlow -> MediPhi-SFT (not part of `microsoft/MediPhi`).** Data: MediFlow, either all 2.5M instructions (CLUE+ 41.9) or the top-quality 800k (CLUE+ 43.0). Table 7: AdamW; linear warmup then cosine; 40 warmup steps; max LR 2e-5; 2 epochs; NEFTune alpha 5; batch size per GPU 16; 8 GPUs; gradient accumulation 2; effective batch size 256. Full vs adapter: not stated.

6. **Stage 2b - DPO on MediFlow-DPO -> MediPhi-Instruct (not part of `microsoft/MediPhi`).** Data: 130,852 triplets with GPT-4o generated rejected outputs. Table 8: AdamW; linear warmup then cosine; 50 warmup steps; max LR 1e-6; 1 epoch; batch size per GPU 8; 16 GPUs; gradient accumulation 1; effective batch size 128; beta 0.1. Table 4 lists "+DPO" under "+SFT 800K" (inference from the table layout: DPO starts from the 800K SFT checkpoint).

### Which checkpoint is `microsoft/MediPhi`?
The card states that the model in this repo was "obtained by merging all 5 experts with the BreadCrumbs technique" (model card, Model Description). So `microsoft/MediPhi` = output of stage 1b-ii = the "MediPhi" row of the paper's tables (CLUE+ 39.3, MedQA 0.491). It has not received MediFlow SFT or DPO. Card metadata: `base_model: microsoft/Phi-3.5-mini-instruct`, tags `merge`, `mergekit`, `medical`, `clinical`, license `mit`; the card also lists "Finetuned from model: microsoft/Phi-3.5-mini-instruct" and model type Phi3.

### The MediPhi family
The card says the collection has 7 models of 3.8B parameters:

| # | Name in card/paper | What distinguishes it | CLUE+ AVG | MedQA (Table 13) |
|---|---|---|---|---|
| 1 | PubMed | expert on PubMed commercial subset + abstracts, SLERP-merged with base | 37.7 | 0.467 |
| 2 | Clinical | expert on open clinical documents (PMC-Patients, NoteChat, Asclepius, mtsamples), SLERP-merged | 39.6 | 0.500 |
| 3 | MedWiki | expert on Medical Wikipedia, SLERP-merged | 39.7 | 0.478 |
| 4 | Guideline | expert on medical guidelines, SLERP-merged | 39.2 | 0.459 |
| 5 | MedCode | expert on medical coding (ICD10CM, ICD10PROC, ICD9CM, ICD9PROC, ATC), SLERP-merged | 36.7 | 0.531 |
| 6 | MediPhi (= `microsoft/MediPhi`) | BreadCrumbs merge of the five experts; the general expert | 39.3 | 0.491 |
| 7 | MediPhi-Instruct | MediPhi + SFT on MediFlow + DPO on MediFlow-DPO; the final aligned model | 43.4 | 0.548 |

- Exact HF repo ids of models 1-5 and 7: not stated in the supplied card (only `microsoft/MediPhi` appears as a repo id; MediPhi-Instruct is named without an org prefix; the dataset is `microsoft/mediflow`).
- The card calls models 1-6 the "six merged MediPhi models".
- Paper-only models, not among the card's 7: MediPhi-SFT (2.5M: CLUE+ 41.9; 800K: CLUE+ 43.0, MedQA 0.536); DataMix (DAPT on all corpora, SLERP-merged, CLUE+ 37.5); unified Task-Arithmetic (39.4) and TIES (39.3) merges; Phi3.5 + SFT 800K (42.2) and + DPO (42.2); the Guideline ablations; ICD10CM case-study models; and a model called "Medical" that appears only in Table 13 (MedQA 0.519) and is not defined (appendix A.1 mentions a "Medical group" covering MedWiki, guidelines and medical coding corpora; (inference) it may be an expert on that combined group).

### Expected structure of the weight delta (inference)
Everything in this subsection is inference from the pipeline above plus background knowledge. None of it is stated by the sources and none of it has been checked against the weights.

1. **Same tensor set, no adapter tensors.** [UNVERIFIED-BACKGROUND] Phi-3-mini architecture as I recall it: hidden 3072, 32 layers, MLP intermediate 8192, vocab 32064, fused `qkv_proj` and `gate_up_proj`, no biases, untied `lm_head`. Per layer: 28,311,552 (qkv_proj) + 9,437,184 (o_proj) + 50,331,648 (gate_up_proj) + 25,165,824 (down_proj) + 6,144 (two norm weights) = 113,252,352; times 32 = 3,624,075,264; plus embed_tokens 98,500,608, lm_head 98,500,608, final norm 3,072 = 3,821,079,552. This equals the parameter count our project measured for both checkpoints, so any adapters (if they were ever used) are already folded into the dense weights. To verify from `config.json` and the safetensors headers.
2. **Which tensors change.** Nothing in the sources says any module was frozen, and a multi-model merge is applied tensor by tensor to the whole model, so the default expectation is that every matrix in all 32 layers (attention `qkv_proj`, `o_proj`; MLP `gate_up_proj`, `down_proj`) differs, probably also `embed_tokens`, `lm_head` and the norm weights. A tensor that is bit-identical to base would be informative (it would indicate a frozen module in all five experts, or a pass-through rule in the merge config).
3. **Low-rank? Not expected.** (a) No adapter is mentioned, and the scale of the continual pre-training (about 48B tokens for PubMed, 16 GPUs, effective batch 512) is typical of full-parameter training. (b) Even if the experts had been LoRA-trained, the delta of `microsoft/MediPhi` is a weighted sum of five expert deltas after element-wise magnitude masking (BreadCrumbs), which destroys exact low-rank structure. So each per-matrix delta should have a broad singular spectrum with no hard rank cutoff. Caveat: a broad spectrum cannot by itself distinguish "full fine-tuning" from "adapters then masking"; only a hard cutoff at rank r (or 5r) would be decisive, and it would contradict this expectation.
4. **Sparse? Probably mostly dense, possibly with a fingerprint of exact zeros.** [UNVERIFIED-BACKGROUND] BreadCrumbs (Davari and Belilovsky 2024, cited by the paper; the title mentions sparse masks) builds a task vector per expert (expert minus base) and, per tensor, masks out both the largest-magnitude entries (outliers) and the smallest-magnitude entries, then adds a weighted sum of the masked task vectors to the base. Each masked task vector is sparse, but the union of five masks is much denser: if each expert keeps a fraction d of entries independently, the fraction of entries with exactly zero delta is about (1-d)^5 (for example 3% at d = 0.5, about 0.001% at d = 0.9). The density is unknown, so the zero fraction could be anywhere from negligible to noticeable. Second signature: the largest per-expert changes are removed, so the distribution of |delta| should have a truncated upper tail compared with a raw fine-tuning delta.
5. **Small relative magnitude, spread over all layers.** For small angles SLERP is close to linear interpolation, so each SLERP expert is roughly base + t x (raw expert - base) with t in {0.10, 0.25, 0.50}; the MediPhi delta is then roughly a weighted sum of masked, down-scaled task vectors. The Guideline ablation (27.2 without SLERP vs base 36.5) suggests the raw task vectors are behaviorally large and the released merge is an attenuated mixture of them. Exact SLERP also adds a tiny component parallel to the base weights (order angle^2 / 8 at t = 0.5), probably below storage precision.
6. **A mixture of five domain directions, not one "medical reasoning" direction.** The delta mixes PubMed, clinical-document, coding, guideline and encyclopedia adaptations. It cannot be decomposed from the two checkpoints alone, but the card says the five SLERP experts are released, so a per-tensor regression of the MediPhi delta on the five expert deltas should be possible.
7. **Small MedQA signal in this particular checkpoint.** The paper's own Table 13 gives MediPhi minus Phi-3.5 = +0.005 on MedQA. Variants with larger reported MedQA differences: MediPhi-Instruct (+0.062), MediPhi-SFT (+0.050), MedCode (+0.045), "Medical" (+0.033), Clinical (+0.014).

**Checks that would confirm or refute the above:**
- A. Compare safetensors headers of both repos: identical key sets and shapes; record dtypes of both (if the stored dtype is a 16-bit float, deltas smaller than half a unit in the last place of the base weight round to exactly zero, which inflates the zero count independently of any mask).
- B. Per tensor: bit-identical or not; max |delta|; relative Frobenius norm of delta vs base; fraction of exactly-zero entries; plot by layer index and module type.
- C. Per 2D tensor: SVD of the delta; singular value spectrum, stable rank, number of singular values for 90% and 99% of the energy. LoRA-style production predicts a hard cutoff; this note predicts none.
- D. Per tensor: histogram of |delta|; look for a truncated upper tail and a gap near zero; test whether exact zeros correlate with large |base weight| (rounding artifact) or not (mask).
- E. Look in the HF repo files for a MergeKit configuration (a YAML file or a YAML block in the README). [UNVERIFIED-BACKGROUND] MergeKit usually writes its config next to the merged weights. The supplied card text contains none. If present, it gives method, weights and densities directly and settles most of the "not stated" items above.
- F. Download the five SLERP experts and MediPhi-Instruct: regress the MediPhi delta on the five expert deltas per tensor (recovers effective weights and masks); estimate each expert's SLERP proportion; check that MediPhi-Instruct minus MediPhi is a small dense delta (SFT at LR 2e-5 for 2 epochs, DPO at LR 1e-6 for 1 epoch).
- G. Layer profile of delta norms: piecewise-constant scale across groups of layers would suggest per-layer-group merge parameters from the evolutionary search ([UNVERIFIED-BACKGROUND] MergeKit's evolutionary tool can optimize parameters per layer group).

## Experiments and numbers
Models: base `Phi-3.5-mini-instruct`; MediPhi family; comparison LLMs Mistral-7B-Instruct-v0.1, BioMistral-7B-DARE, Meta-Llama-3-8B-Instruct, Llama3-Med42-8B. Benchmark: CLUE+ (12 datasets, see next section for per-dataset numbers and protocol). No seeds, variances or confidence intervals are reported anywhere.

**ICD10CM case study (Figures 4 and 5; accuracy on MedConceptsQA ICD10CM, easy and medium questions left out).**

| Model | Accuracy | After SLERP 50% with base |
|---|---|---|
| GPT-4o-0806 | 0.71 | - |
| GPT-4-0125 | 0.57 | - |
| Phi3.5-mini-instruct | 0.45 | - |
| DAPT on webpages | 0.25 | not shown |
| Explainer | 0.51 | 0.58 |
| PIT QA | 0.43 | 0.62 |
| PIT Summary | 0.53 | 0.65 |
| PIT Entities | 0.47 | 0.63 |
| PIT Relations | 0.45 | 0.64 |

The abstract's "44% on ICD-10 coding (outperforming GPT-4-0125 by 14%)" refers to 0.65 vs 0.45 and 0.65 vs 0.57 (relative).

**Table 2 - SLERP-merged experts, CLUE+ average.**

| Group | Model | AVG | #DG | CV Delta |
|---|---|---|---|---|
| Baseline | Phi-3.5 mini | 36.5 | - | - |
| Baseline | DataMix | 37.5 | 10 | 1.2 |
| SLERP experts | PubMed | 37.7 | 9 | 1.8 |
| SLERP experts | Clinical | 39.6 | 10 | 2.0 |
| SLERP experts | MedWiki | 39.7 | 10 | 1.5 |
| SLERP experts | MedCode | 36.7 | 5 | 39.6 |
| SLERP experts | Guideline | 39.2 | 10 | 1.8 |
| Guideline ablations | w/o SLERP | 27.2 | 4 | 1.0 |
| Guideline ablations | w/o PIT | 33.0 | 6 | 0.9 |
| Guideline ablations | w/o SLERP&PIT | 25.2 | 1 | 2.8 |

**Table 3 - unifying the five experts, CLUE+ average.**

| Group | Model | AVG | #DG | CV Delta |
|---|---|---|---|---|
| Baseline | Phi-3.5 mini | 36.5 | - | - |
| SLERP experts (per-dataset statistics) | Minimum | 34.5 | 4 | 1.9 |
| SLERP experts (per-dataset statistics) | Average | 38.5 | 7 | 2.1 |
| SLERP experts (per-dataset statistics) | Maximum | 43.1 | 11 | 1.1 |
| SLERP experts | DataMix | 37.5 | 10 | 1.2 |
| Unified SLM | Task-Arithmetic | 39.4 | 9 | 1.9 |
| Unified SLM | Ties | 39.3 | 7 | 1.7 |
| Unified SLM | BreadCrumbs | 39.3 | 11 | 1.5 |

**Table 4 - alignment with MediFlow, CLUE+ average.**

| Start | Step | AVG | #DG | CV Delta |
|---|---|---|---|---|
| Phi3.5 mini | - | 36.5 | - | - |
| Phi3.5 mini | +SFT 800K | 42.2 | 9 | 1.4 |
| Phi3.5 mini | +DPO | 42.2 | 8 | 1.4 |
| MediPhi | - | 39.3 | 11 | 1.5 |
| MediPhi | +SFT 2.5M | 41.9 | 9 | 1.6 |
| MediPhi | +SFT 800K | 43.0 | 9 | 1.4 |
| MediPhi | +DPO | 43.4 | 9 | 1.4 |

**Table 5 - comparison with other medical LLMs, CLUE+ average.**

| Model | AVG (gain over its base) | #DG | CV Delta |
|---|---|---|---|
| Mistral-7B-Instruct-v0.1 | 33.6 | - | - |
| BioMistral-7B-DARE | 34.7 (+1.1) | 8 | 3.4 |
| Phi-3.5 mini (3.8B) | 36.5 | - | - |
| MediPhi (3.8B) | 39.3 (+2.8) | 11 | 1.5 |
| MediPhi-SFT (3.8B) | 43.0 (+6.5) | 9 | 1.4 |
| MediPhi-Instruct (3.8B) | 43.4 (+6.9) | 9 | 1.4 |
| Meta-Llama-3-8B-Instruct | 44.1 | - | - |
| Llama3-Med42-8B (fine-tuned on ACI-Bench, so not few-shot there) | 45.3 (+1.2) | 5 | 7.8 |

Other reported figures: MediPhi-Instruct relative gains over base of 64.3% on SDoH and 49.5% on RRS QA; BreadCrumbs improves 10.6% over the MedWiki expert on ICD10CM while the MedCode expert improves 36.9% over MedWiki there; aligned models lose on Problem List Summarization and MeDiSumCode, which the authors attribute to a bias of MediFlow towards listing tasks. Compute: 8x80GB A100 GPUs on Azure ML, approximately 12,000 GPU-hours in total, close to 3,600 GPU-hours for the final model with evaluations; close to 25B input-output tokens of Azure OpenAI usage (GPT-4o, GPT-4o mini, text-embedding-3-large).

## MedQA and other evaluation details

### MedQA protocol (what the sources give)
| Item | Sources |
|---|---|
| Where reported | Paper appendix Table 13 only (multiple-choice medical QA benchmarks). The table is never referenced or discussed in the body text. The model card does not report MedQA at all. |
| MedQA variant (USMLE 4-option vs 5-option, language) | not stated (dataset cited as Jin et al. 2021) |
| Split and number of test questions | not stated |
| Prompt format, system prompt | not stated |
| Zero-shot vs few-shot | not stated |
| Answer extraction (generated letter parsing vs option log-likelihood) | not stated |
| Decoding settings | not stated for these benchmarks (greedy decoding is stated only for the six new CLUE+ datasets, Table 9) |
| Metric | values are on a 0-1 scale with an AVG column; the metric name is not stated (inference: accuracy) |
| Seeds, variance, confidence intervals | not stated |
| Evaluation harness | not stated |

**Table 13 - multiple-choice medical QA benchmarks (exact values).**

| Model | MedQA | MedMCQA | PubMedQA | MMLU-med | AVG |
|---|---|---|---|---|---|
| Phi-3.5-mini-instruct | 0.486 | 0.554 | 0.768 | 0.715 | 0.631 |
| PubMed | 0.467 | 0.549 | 0.774 | 0.724 | 0.629 |
| Clinical | 0.500 | 0.551 | 0.772 | 0.727 | 0.638 |
| Medical | 0.519 | 0.535 | 0.740 | 0.701 | 0.624 |
| MedWiki | 0.478 | 0.548 | 0.768 | 0.719 | 0.628 |
| MedCode | 0.531 | 0.532 | 0.758 | 0.700 | 0.630 |
| Guideline | 0.459 | 0.556 | 0.770 | 0.724 | 0.627 |
| MediPhi | 0.491 | 0.559 | 0.766 | 0.720 | 0.634 |
| MediPhi-SFT | 0.536 | 0.552 | 0.766 | 0.716 | 0.642 |
| MediPhi-Instruct | 0.548 | 0.555 | 0.764 | 0.714 | 0.645 |

Differences vs Phi-3.5-mini-instruct (my arithmetic from the table):

| Model | MedQA | MedMCQA | PubMedQA | MMLU-med | AVG |
|---|---|---|---|---|---|
| MediPhi (= `microsoft/MediPhi`) | +0.005 | +0.005 | -0.002 | +0.005 | +0.003 |
| MediPhi-SFT | +0.050 | -0.002 | -0.002 | +0.001 | +0.011 |
| MediPhi-Instruct | +0.062 | +0.001 | -0.004 | -0.001 | +0.014 |
| MedCode | +0.045 | -0.022 | -0.010 | -0.015 | -0.001 |

Notes for checking our own evaluation:
- The protocol cannot be reproduced from the sources, so a mismatch between our absolute accuracies and 0.486 / 0.491 is not by itself evidence of a bug on our side. What can be compared is the sign and rough size of the gap.
- (inference) The reported net MedQA gain of `microsoft/MediPhi` over the base is half a point. The set "MediPhi right, Phi wrong" may therefore be nearly offset by the set "Phi right, MediPhi wrong".
- [UNVERIFIED-BACKGROUND] The commonly used MedQA-USMLE 4-option test split has 1,273 questions. If that split was used, 0.486 and 0.491 correspond to about 619 and 625 correct, a net difference of about 6 questions. The three-decimal values are compatible with n = 1,273 but do not identify it.
- Training-data overlap: the stated training corpora do not include the MedQA training set (the paper criticizes other medical models for training on benchmark training sets), but the paper does not explicitly state a no-overlap check. MediFlow includes multiple-choice QA as one of its 14 task types, and the merge was tuned on synthetic multiple-choice validation sets.

### Is the MedQA prompt released anywhere? (checked 2026-11-23)
All sources are public and were fetched without logging in.

| Where | What is there | MedQA prompt or evaluation code? |
|---|---|---|
| arXiv:2505.10717v2 (the latest version), full text with appendices | Table 13 with its caption only. System prompts are printed for MEDIQA-RRS QA, ACI-Bench, MedicationQA, MEDEC and SDoH. For the six CLUE datasets: "the same configuration (i.e. prompts, metrics, and few shots) as Dada et al. (2024)" | no |
| ACL 2025 camera-ready (`2025.acl-long.950`, 23 pages) | The same content. The only attachment on the Anthology page is a video | no |
| Hugging Face `microsoft/MediPhi` (last modified 2025-12-15) | Weights, tokenizer, config, `mergekit_config.yml`, README. The card reports CLUE+ only. No discussion threads | no |
| Hugging Face `microsoft/MediPhi-Instruct` (last modified 2025-12-15) | Weights, tokenizer, config, README. The card reports CLUE+ only. Five discussion threads, none about evaluation | no |
| Hugging Face dataset `microsoft/mediflow` | Training data files and a figure | no |
| GitHub `TIO-IKIM/CLUE` (the CLUE benchmark code; 43 files) | The six clinical CLUE tasks. No MedQA, no MultiMedQA, no lm-evaluation-harness | no |
| CLUE paper, arXiv:2404.04067 | Uses multiple-choice numbers "reported" by other work, averaged over MedMCQA, MedQA and PubMedQA | no |
| GitHub repository search for "MediPhi" | No repository from the authors | no |

- So the MedQA prompt behind Table 13 is not released in any of these places. Only the authors can confirm it
  (the model card gives a contact address).
- (inference, ours) Table 13 was produced with the `multimedqa` task group of lm-evaluation-harness: its four
  columns match that group, and our re-implementation of the group's `medqa_4options` task reproduces the three
  public models' MedQA values to within 1 to 13 questions of 1,273 (EXP-04 Protocol C). The other three columns
  were not replicated.
- The model card's "Input Format" puts the instruction in the system role and the document in the user role
  (`<|system|> ... <|end|> <|user|> ... <|end|> <|assistant|>`). Our chat prompt puts the instruction and the
  question in one user turn with no system message. A system-role variant has not been scored. (inference) If
  MediPhi-Instruct depends on that format, our chat-prompt comparison could understate it.

### CLUE+ protocol
- CLUE subset (MedNLI, MeQSum, Problem List Summarization, LongHealth, MeDiSumQA, MeDiSumCode): same prompts, metrics and few shots as Dada et al. 2024 (details not reproduced in this paper).
- New CLUE+ subset (Table 9), decoding = greedy:

| Dataset | Few-shots | Metric |
|---|---|---|
| ICD10CM (MedConceptsQA) | 3 | AVG accuracy |
| MedicationQA | 3 | Rouge-1 F1 |
| RRS QA (MEDIQA-RRS reformulated as QA) | 3 | Rouge-1 F1 |
| SDoH | 4 | Type-match F1 with boundary overlap |
| ACI-Bench | 1 | Rouge-1 F1 |
| MEDEC | 2 | Sentence ID accuracy |

- Dataset preparation: MedicationQA 674 questions filtered and answers reformulated with Llama-3.3-70B-Instruct to 485 QA pairs; MEDIQA-RRS test split of 600 finding-impression pairs with impressions reformulated into QA pairs by the same LLM. Dataset-specific system prompts are given in appendix A.8.
- Observation: the base scores 0.45 on ICD10CM in Figures 4-5 but 49.3 in the CLUE+ ICD10CM column, so the two ICD10CM setups differ (difference not explained; the CLUE+ column uses 3 shots).

**CLUE+ per-dataset results (paper Tables 11 and 12 combined; AVG from Tables 2-5).** PLS = Problem List Summary, LH = LongHealth, ACI = ACI-Bench.

| Model | MedNLI | PLS | MeQSum | LH | MeDiSumQA | MeDiSumCode | RRS QA | MedicationQA | MEDEC | ACI | SDoH | ICD10CM | AVG |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Phi-3.5-mini-instruct | 66.6 | 28.4 | 36.7 | 45.9 | 25.9 | 41.1 | 41.2 | 11.2 | 14.8 | 42.3 | 35.1 | 49.3 | 36.5 |
| DataMix | 68.5 | 29.0 | 37.7 | 45.7 | 26.6 | 41.4 | 43.3 | 10.8 | 18.8 | 42.7 | 36.2 | 49.5 | 37.5 |
| PubMed | 68.3 | 29.2 | 37.6 | 45.7 | 26.3 | 41.0 | 44.1 | 10.3 | 22.2 | 42.7 | 35.8 | 49.5 | 37.7 |
| Clinical | 69.2 | 29.4 | 38.1 | 43.5 | 26.7 | 40.5 | 52.1 | 12.0 | 34.5 | 43.9 | 35.8 | 49.6 | 39.6 |
| MedWiki | 72.8 | 29.2 | 37.6 | 43.6 | 25.1 | 41.7 | 46.7 | 12.2 | 28.8 | 44.7 | 43.6 | 50.2 | 39.7 |
| MedCode | 68.5 | 22.3 | 33.5 | 45.7 | 23.6 | 39.0 | 45.6 | 12.0 | 18.1 | 39.0 | 24.8 | 68.7 | 36.7 |
| Guideline | 70.3 | 29.8 | 37.6 | 41.1 | 25.1 | 41.9 | 48.9 | 11.9 | 28.3 | 44.7 | 41.0 | 49.8 | 39.2 |
| MediPhi (BreadCrumbs) | 66.9 | 28.8 | 37.9 | 45.7 | 26.1 | 41.7 | 44.5 | 11.3 | 29.1 | 44.3 | 39.7 | 55.5 | 39.3 |
| MediPhi-SFT | 70.6 | 26.9 | 42.8 | 44.2 | 28.8 | 35.0 | 60.8 | 18.8 | 35.0 | 43.4 | 54.5 | 54.9 | 43.0 |
| MediPhi-Instruct | 71.0 | 26.0 | 42.8 | 45.0 | 29.1 | 37.2 | 61.6 | 19.3 | 34.4 | 43.5 | 56.7 | 54.9 | 43.4 |
| Mistral-7B-Instruct-v0.1 | 64.8 | 25.0 | 31.1 | 30.0 | 25.5 | 13.9 | 50.4 | 22.7 | 21.5 | 50.4 | 40.2 | 27.6 | 33.6 |
| BioMistral-7B-DARE | 66.8 | 28.4 | 34.5 | 30.5 | 25.7 | 21.3 | 49.6 | 22.3 | 23.1 | 43.3 | 45.9 | 25.1 | 34.7 |
| Meta-Llama-3-8B-Instruct | 74.1 | 31.6 | 39.5 | 58.8 | 30.3 | 27.8 | 55.8 | 26.1 | 46.5 | 50.2 | 63.1 | 25.7 | 44.1 |
| Llama3-Med42-8B | 77.5 | 32.4 | 42.8 | 57.9 | 29.7 | 25.2 | 54.1 | 25.7 | 35.4 | 56.5 | 53.9 | 53.4 | 45.3 |

The model card's CLUE+ table (columns Phi-3.5-mini-instruct, PubMed, Clinical, MedWiki, Guideline, MedCode, MediPhi, MediPhi-Instruct) carries the same per-dataset values and the averages 36.5, 37.7, 39.6, 39.7, 39.2, 36.7, 39.3, 43.4.

### Model card statements relevant to a research replication
- **License:** MIT (card front matter and License section; the license link points to the Phi-3.5-mini-instruct LICENSE file). The paper's footnote on the "commercially permissive license" is still a "TBD" placeholder in arXiv v2.
- **Intended use:** research use in English, especially clinical NLP; medically adapted models for memory/compute-constrained and latency-bound scenarios. The card says the model should be used for research purposes, in a benchmarking context or with expert verification of outputs. It is not specifically designed or evaluated for all downstream purposes; developers must evaluate and mitigate accuracy, safety and fairness issues before a downstream use, particularly high-risk ones, and follow applicable laws. The card adds that nothing in it restricts or modifies the license.
- **Paper's own recommendation (Limitations):** use the MediPhi collection specifically on clinical NLP tasks, with verification by a domain expert; effects on abilities outside the medical/clinical domain and instruction following were not studied.
- **Safety:** the card lists the usual Phi-family responsible-AI caveats (quality of service across languages, stereotypes, inappropriate content, information reliability, limited code scope, long-conversation drift) and high-risk cautions (health or legal advice needs application-level safeguards). Red teaming: the card cites a separate medical red-teaming protocol (arXiv 2507.07248) and states that all six merged MediPhi models fully conserve the base model's safety capabilities, and that MediPhi-Instruct conserved behavior on jailbreaking and harmfulness and improved on groundedness. It also points to a separate clinical information-extraction benchmark (arXiv 2507.05517). Neither of those two works is among our supplied sources.
- **Recommended input format (card, "Input Format"):**

```
<|system|>
GIVE A ROLE AND INSTRUCTIONS FOR CLINICAL NLPTASKS<|end|>
<|user|>
INPUT DOCUMENT<|end|>
<|assistant|>
```

- **Card usage example:** a `messages` list with a `system` turn (role and task instructions) and a `user` turn (the input document), passed to a transformers `text-generation` pipeline; generation arguments `max_new_tokens: 500`, `return_full_text: False`, `temperature: 0.0`, `do_sample: False`; `torch.random.manual_seed(0)`; loading with `device_map="cuda"`, `torch_dtype="auto"`, `trust_remote_code=True`; optional `attn_implementation="flash_attention_2"`, or `"eager"` on older GPUs. The card says Phi-3.5-mini is best suited to chat-format prompts but plain text is possible, and refers to `microsoft/Phi-3.5-mini-instruct` for tokenizer details.
- (inference) For our MedQA runs the card-consistent setup is: Phi-3 chat template with a system instruction plus the question as the user turn, greedy decoding, identical for both checkpoints. The card gives no MedQA-specific prompt.

## What it establishes
- Under the authors' CLUE+ protocol, each SLERP-merged expert averages above the base (36.7 to 39.7 vs 36.5), the BreadCrumbs merge reaches 39.3 with gains on 11 of 12 datasets, and SFT plus DPO on MediFlow reaches 43.4.
- On one corpus group (Guideline), both PIT and merging back into the base are needed: 39.2 with both vs 27.2 without SLERP, 33.0 without PIT, 25.2 without both.
- On hard ICD10CM questions, PIT with summaries followed by SLERP 50% gives 0.65 vs 0.45 for the base and 0.57 for GPT-4-0125.
- SFT on the 800k top-quality MediFlow subset beats SFT on all 2.5M (43.0 vs 41.9); applying the same SFT directly to Phi-3.5 gives 42.2.
- On multiple-choice medical QA (Table 13) the average moves little (0.631 -> 0.634 -> 0.645); MedQA is 0.486 (base), 0.491 (MediPhi), 0.548 (MediPhi-Instruct).
- Per the card, the released `microsoft/MediPhi` is the BreadCrumbs merge of five experts.

## What it does NOT establish
- How the experts were parameterized during training (full fine-tuning vs adapters) and the merge hyperparameters. The structure of the weight delta is therefore not established by either source.
- That `microsoft/MediPhi` is better than Phi-3.5 on MedQA in any robust sense: the reported difference is 0.005, with no protocol, variance or significance test.
- Statistical reliability of any number: no seeds, repeats or confidence intervals.
- Which layers or components carry the gains: there is no mechanistic or interpretability analysis.
- Preservation of general abilities (multilingual, conversational, safety): the paper says the impact outside medical/clinical and instruction-following abilities was not studied; the card's safety claims rest on separate work.
- A broad improvement in medical knowledge: MedMCQA, PubMedQA and MMLU-med are flat or slightly lower for several variants, and individual experts lose on some datasets (for example MedCode on PLS, MeQSum, SDoH).
- That the expert-plus-merge stage is necessary for most of the final gain: Phi-3.5 with SFT alone reaches 42.2 vs 43.0 for MediPhi with the same SFT.
- The "44% on ICD-10 coding" figure does not describe the released MediPhi (55.5 vs 49.3 on the CLUE+ ICD10CM column); it describes the ICD10CM case-study model.

## Limitations stated by the authors
- Large resource needs: 8x80GB A100 GPUs, approximately 12,000 GPU-hours, plus close to 25B tokens of Azure OpenAI usage.
- MediFlow is single-turn only, has few very-high-complexity tasks with long inputs/outputs, and is English-only.
- Abilities inherited from Phi-3.5 (multilingual, conversational, safety alignment) may be affected; not studied beyond medical/clinical and instruction-following abilities. Recommended use: clinical NLP tasks, with expert verification of outputs.
- CLUE+ gaps: information extraction is represented only by SDoH; no nursing-domain inputs.
- Exact reproduction may become impossible as GPT-4o versions change and because of their stochasticity.
- Ethics statement: compute and API usage contribute to carbon emissions and limit reproduction.

## Implementation notes for med_wsp
- **Artifacts named by the sources:** model `microsoft/MediPhi` (card); dataset `microsoft/mediflow` (card); six other models in the collection (repo ids not stated); synthetic merge-validation sets said to be released (location not stated). No code repository link in either source. Training code: HF transformers/trl/accelerate/datasets; merging: MergeKit.
- **Compute reported:** approximately 12,000 A100 GPU-hours overall. Not reproducible locally.
- **36 GB Apple M4 Max (inference, estimates):**
  - Fits: both checkpoints in 16-bit floats (3,821,079,552 x 2 bytes is about 7.6 GB each); MedQA inference for both models; per-tensor weight diff statistics and SVDs done one tensor at a time (largest matrices: 16384 x 3072 `gate_up_proj` and 32064 x 3072 embeddings, see the parameter arithmetic above); activation patching with both models loaded in 16-bit (about 15.3 GB of weights plus activations).
  - Tight: both checkpoints in float32 at once (about 15.3 GB each); prefer lazy per-tensor loading.
  - Does not fit: any retraining stage (PubMed DAPT on about 48B tokens, PIT on GPT-4o generated data, SFT on 800k-2.5M instructions), regenerating MediFlow, or the 500-evaluation evolutionary merge search at useful speed.
- **Gotchas:**
  - `microsoft/MediPhi` is the merge, not the instruct-aligned model. If the experiments need a large MedQA gap, the paper's numbers point to MediPhi-Instruct (0.548) or the MedCode expert (0.531) rather than MediPhi (0.491).
  - The MedQA protocol behind Table 13 is unknown; fix our own protocol and apply it identically to both checkpoints.
  - The card's example targets CUDA and mentions flash attention; on Apple Silicon use the standard attention implementation (inference).
  - Inconsistencies in the paper to keep in mind: DAPT LR 5e-5 in the text vs 1e-4 in Table 6; Table 2 caption says experts were obtained with PIT on all five groups while Sec. 3.1.2 says PubMed uses DAPT; "8x80GB A100" in Limitations vs 16 GPUs in Tables 6 and 8; five difficulty levels in appendix A.4 vs six in the introduction and in the generation prompt; judge criteria named differently in Sec. 3.2.1 and appendix A.5; the DPO set is called MediFlow-DPO in the text and MediPhi-DPO in appendix A.7; SDoH relative gain reported as 64.3% while Table 12 values (56.7 vs 35.1) give 61.5% by my arithmetic.

## Relevance to med_wsp (provisional)
All items below are provisional ideas for the researcher, not decisions.
- This paper and card define the model pair we compare. The key correction to our working picture: the difference between the two checkpoints is a multi-expert model merge (SLERP then BreadCrumbs) of continually pre-trained experts, not a single fine-tuning run and (as far as the sources say) not merged LoRA adapters.
- The reported MedQA gap for exactly our pair is +0.005. If our own evaluation confirms a near-zero net gap, the "MediPhi right, Phi wrong" subset still exists but must be analysed together with the reverse subset, and it may be worth adding MediPhi-Instruct or individual experts (MedCode, Clinical) as additional comparison points.
- For weight-diff analysis and [[Weight-space programming]] style edits: expect dense, high-rank per-matrix deltas spread over all layers (inference, see checks A-G). Low-rank approximations of the delta would be approximations to test, not a recovery of an underlying adapter. The released experts would give a natural five-way decomposition of the delta by domain.
- For [[Activation patching]]: the two models share architecture, tokenizer and parameter count, so patching between them is well defined; the delta mixes domain knowledge with whatever the merge kept of PIT-induced changes, so component-level attributions may not be purely "medical reasoning".
- For [[Crosscoders]] and [[Transcoders]] between base and fine-tune: the merge keeps MediPhi close to the base by construction (SLERP attenuation), which is relevant when deciding whether dictionaries trained on the base can be reused or adapted (see [[Tan et al. 2026 - Efficient Transcoder Adaptation]], [[Lindsey et al. 2024 - Sparse Crosscoders]]).
- For targeted transfer into the base without fine-tuning: adding the whole delta trivially reproduces MediPhi, so the open question for our experiments is which subset of tensors, layers or directions of the delta is sufficient for the MedQA-relevant behavior.
- For agent-driven circuit discovery and [[Attribution graphs]] ([[Ameisen et al. 2025 - Circuit Tracing]]): no direct input from this paper beyond the model definitions and the evaluation caveats.

## Open questions
- Were the experts trained with full-parameter fine-tuning or with adapters? Which modules were trainable (embeddings, lm_head, norms)?
- What are the SLERP proportions per expert and the BreadCrumbs weights, density and outlier fraction? Is there a MergeKit config in the HF repo?
- Were the BreadCrumbs inputs the raw experts or the SLERP-merged experts?
- What MedQA protocol produced Table 13 (variant, split size, shots, prompt, answer extraction, decoding)? Does our harness land near 0.486 and 0.491?
- Is the +0.005 MedQA difference within evaluation noise?
- What is the "Medical" model in Table 13?
- Which learning rate did the PubMed expert actually use (5e-5 or 1e-4)? Did the final MedCode expert use explainers, summaries or both?
- What dtype are the released MediPhi weights stored in, and does rounding create artificial exact zeros in the delta?
- What are the HF repo ids of the five experts and of MediPhi-Instruct?
- The text read is arXiv v2; does the ACL camera-ready version add the MCQ protocol or merge details?

## Ledger card
**Full citation:** Corbeil, J.-P., Dada, A., Attendu, J.-M., Ben Abacha, A., Sordoni, A., Caccia, L., Beaulieu, F., Lin, T., Kleesiek, J., Vozila, P. (2025). A Modular Approach for Clinical SLMs Driven by Synthetic Data with Pre-Instruction Tuning, Model Merging, and Clinical-Tasks Alignment. Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 19352-19374, Vienna, Austria. (Venue details from the model card's BibTeX; text read is arXiv:2505.10717v2, 21 May 2025.)
**Source / DOI / URL:** https://arxiv.org/abs/2505.10717 ; DOI 10.18653/v1/2025.acl-long.950 ; https://aclanthology.org/2025.acl-long.950/ ; model card of `microsoft/MediPhi` (local copy `papers/pdf/mediphi_model_card.md`).
**Why I read it:** It defines how MediPhi's weights were produced from Phi-3.5-mini-instruct and reports the MedQA numbers for both, which our project needs before interpreting the weight difference.
**Problem addressed:** Adapting a 3.8B SLM to medical and clinical NLP without access to real clinical data, while keeping the base model's instruction-following abilities.
**Method / representation:** Continual pre-training of five domain experts (PIT, DAPT), SLERP merge of each expert with the base, BreadCrumbs merge of the five experts (MediPhi), then SFT and DPO on synthetic MediFlow data (MediPhi-Instruct).
**Assumptions:** GPT-4o generated task data, validation sets and instructions are of sufficient quality; synthetic multiple-choice validation accuracy is a usable proxy for selecting merge configurations; CLUE+ few-shot scores measure clinical ability.
**Evidence used by the paper:** experiments: CLUE+ averages and per-dataset scores for experts, merges and aligned models; ICD10CM case study; one Guideline ablation; Table 13 multiple-choice benchmarks / analysis: #DG and CV Delta summaries, no significance tests / datasets or systems: PubMed, PMC-Patients, Asclepius, NoteChat, MTSamples, ICD/ATC pages, guidelines, MedWiki, MediFlow; Phi-3.5-mini-instruct, Mistral-7B, Llama-3-8B families.
**Safe citation claims (our wording):** "MediPhi is a collection of 3.8B-parameter models derived from Phi-3.5-mini-instruct by continual pre-training of domain experts, model merging (SLERP, BreadCrumbs) and, for MediPhi-Instruct, SFT and DPO on the synthetic MediFlow dataset"; "The released microsoft/MediPhi checkpoint is, per its model card, the BreadCrumbs merge of five domain experts"; "The authors report CLUE+ averages of 36.5 for Phi-3.5-mini-instruct, 39.3 for MediPhi and 43.4 for MediPhi-Instruct"; "The authors report MedQA scores of 0.486 (Phi-3.5-mini-instruct), 0.491 (MediPhi) and 0.548 (MediPhi-Instruct) without describing the evaluation protocol".
**Unsafe / unsupported citation uses:** "MediPhi was produced by LoRA (or by full) fine-tuning" (not stated); "MediPhi substantially improves MedQA over Phi-3.5" for `microsoft/MediPhi` (reported +0.005; +0.062 is MediPhi-Instruct); attributing a specific MedQA protocol (zero-shot, 4-option, 1,273 questions) to the paper; "the MediPhi weight delta is sparse / low-rank / dense" as a claim of the paper (our inference only); "MediPhi preserves general capabilities" as an established result of this paper.
**Relation to our project:** supporting source (defines the model pair under study); background for model merging.
**Verification status:** full text read: yes (arXiv v2 extracted text incl. appendices; tables, equations and figures cross-checked on PDF pages 1-9, 13-15, 21-23); key claim checked: yes for reported numbers against the PDF; no for all weight-structure inferences (not yet checked against the actual checkpoints).

## Links
- Project concepts: [[Weight-space programming]], [[Activation patching]], [[Crosscoders]], [[Transcoders]], [[Attribution graphs]], [[Layer looping]]
- Weight-space methods: [[Takehi et al. 2026 - Weight-Space Programming]], [[Li et al. 2026 - WeightCode]]
- Base-vs-fine-tune feature comparison: [[Lindsey et al. 2024 - Sparse Crosscoders]], [[Tan et al. 2026 - Efficient Transcoder Adaptation]]
- Circuit discovery: [[Ameisen et al. 2025 - Circuit Tracing]]
- Looped layers: [[Chen et al. 2026 - Training-Free Looped Transformers]], [[Lys et al. 2026 - Inner Loop Inference]]
- This note: [[Corbeil et al. 2025 - MediPhi]]
