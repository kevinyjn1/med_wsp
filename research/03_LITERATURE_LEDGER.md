# Literature Ledger

> **Purpose:** Evidence-oriented record of important papers.  
> **Rule:** Record what each source establishes and does not establish. Do not use this file to declare novelty.

## How this file is maintained

Cards are copied from the "Ledger card" section of each paper note in `vault/papers/` (written by reader agents from the full text on 2026-11-21/22; not yet reviewed by the researcher). Each note separates what the source establishes from what it does not.

### `Ameisen et al. 2025` - Circuit Tracing

Full note: `vault/papers/Ameisen et al. 2025 - Circuit Tracing.md`

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

---

### `Chen et al. 2026` - Training-Free Looped Transformers

Full note: `vault/papers/Chen et al. 2026 - Training-Free Looped Transformers.md`

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

---

### `Corbeil et al. 2025` - MediPhi

Full note: `vault/papers/Corbeil et al. 2025 - MediPhi.md`

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

---

### `Li et al. 2026` - WeightCode

Full note: `vault/papers/Li et al. 2026 - WeightCode.md`

**Full citation:** Bo Li, Taoyong Cui, Bob Zhang, Mengran Li, Chengyang Zhang, Yang Zhang, Pheng Ann Heng, Wanli Ouyang, Zhenfei Yin, Yingcheng Wu, Ling Yang. "WeightCode: Installing New Capabilities into Neural Networks via Weight-Space Programming." arXiv:2611.25476v1 [cs.LG], 19 Nov 2026.
**Source / DOI / URL:** https://arxiv.org/abs/2611.25476 ; code https://github.com/Gen-Verse/WeightCode ; local PDF `papers/pdf/arxiv_2611.25476.pdf` (26 pages).
**Why I read it:** closest paper-form treatment of weight-space programming as a method for installing capabilities; needed to know whether it offers a technique usable on Phi-3.5.
**Problem addressed:** installing a new capability into a trained network from rules, demonstrations or a few labeled examples, without gradient training, while keeping old behavior.
**Method / representation:** frozen LLM writes an executable NumPy weight program $W' = p(W, I)$; candidates selected by 8 labeled checks under a retention rule; program kept and rerun on other networks.
**Assumptions:** the capability can be connected to computation the network already performs; a check split with labels exists; low-use capacity exists to host new gates; edits limited to 20% of weights.
**Evidence used by the paper:** experiments: SCAN (2 splits), BabyAI, CIFAR-10-C on ViT-Lite / ResNet-18 / ResNet-20, ImageNet-C on ResNet-18, reuse on 5 seeds, 12 second-round sessions / analysis: causal interventions on gates, alternative constructions, cross-corruption effects / datasets or systems: SCAN, BabyAI, CIFAR-10, CIFAR-10-C, ImageNet, ImageNet-C, ImageNetV2, CIFAR-100 (CEC protocol).
**Safe citation claims:** "A frozen LLM can write gradient-free weight programs that install robustness or new behaviors in small trained networks while a retention rule bounds loss of old behavior (Li et al. 2026)."; "Rerunning a weight program on independently trained ViT-Lites kept 83% to 110% of its source gain, whereas copying the fixed edit gave -1.42 to +0.01 points on average (Li et al. 2026)."; "On SCAN add-primitive a 257-weight edit raised exact match from 0.0% to 99.8% (Li et al. 2026)."
**Unsafe / unsupported citation uses:** "WeightCode edits large language models" (it does not); "weight programs outperform fine-tuning in general" (only under matched 64 + 64 examples, 8 checks and a tight retention bound; fine-tuning overtakes with more data or a looser bound); "weight programs can add capabilities the network has no representation for" (Sec. A.10 shows the opposite tendency).
**Relation to our project:** closest prior art | overlapping method (protocol design); background for scale (no LLM-scale evidence).
**Verification status:** full text read: yes; key claim checked: yes (numbers checked against Tables 2, 3, 7, 8, 10, 11, 13 in the extracted text; Eq. 2 checked against PDF page 6).

---

### `Lindsey et al. 2024` - Sparse Crosscoders

Full note: `vault/papers/Lindsey et al. 2024 - Sparse Crosscoders.md`

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

---

### `Lys et al. 2026` - Inner Loop Inference

Full note: `vault/papers/Lys et al. 2026 - Inner Loop Inference.md`

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

---

### `Takehi et al. 2026` - Weight-Space Programming

Full note: `vault/papers/Takehi et al. 2026 - Weight-Space Programming.md`

**Full citation:** Rikiya Takehi, Jayden Teoh, Alex Zhang, and Omar Khattab. "Sparks of Weight-Space Programming: LLMs can natively read & write neural network weights." Blog post, MIT CSAIL, November 2026.
**Source / DOI / URL:** https://rikiyat.github.io/blog/weight-space-programming/ ; logs github.com/RikiyaT/lm_understand-logs (local clone `external/lm_understand-logs/`).
**Why I read it:** primary source for the replication targets (experiments 3, 7, 8) and for the concept the project builds on.
**Problem addressed:** whether a general-purpose LLM can read, fix and write neural network weights from a natural-language goal without gradients.
**Method / representation:** `gpt-6-astra` via Codex CLI with all native tools off; weights as text or via a Python sandbox; budgets on code runs and forward passes; direct edits saved through a harness API.
**Assumptions:** the harness rules (no forward pass in the sandbox, no training) are enforced by prompt plus audit of every code run; evaluation benchmarks are never shown to the model.
**Evidence used by the paper:** experiments: eight experiments plus a Gemma follow-up, each with released logs / analysis: per-experiment READMEs mapping every number to a file / datasets or systems: own grid games, MinAtar, Kinetix, 9x9 Go (pgx), char-LMs on Python / romaji / English, Gemma 3 270M, Qwen3.5-0.8B, GSM8K, MATH, MATH-500, IFEval, MMLU-Pro, GPQA Diamond, AMC23, Minerva Math, OlympiadBench.
**Safe citation claims:** "Takehi et al. (2026) report that an LLM-chosen edit of 131,072 weights (0.016%) of Qwen3.5-0.8B raised MATH-500 from 7.9% to 48.1% under thinking-on sampling by suppressing reasoning loops."; "In their experiment 7 the LLM could not identify the sycophantic fine-tune from weights alone in four attempts, but did so for three of three pairs when given attention patterns."; "The authors present Weight-Space Programming as a concept, not a method."
**Unsafe / unsupported citation uses:** "LLMs can improve the math ability of language models by editing weights" (the Gemma follow-up shows no gain; the Qwen gain is loop removal); "Astra located the sycophancy circuit" (no causal verification; patch check closed 0.00 of the gap); "weight-space programming beats PPO" (baselines are not tuned comparisons; Seaquest and Asterix are below PPO at 500M steps).
**Relation to our project:** closest prior art | overlapping method (we replicate experiments 3, 7, 8 and reuse the harness pattern).
**Verification status:** full text read: yes; key claim checked: yes (blog numbers cross-checked against `exp3_language/README.md`, `exp7_sycophancy/README.md`, `exp8_math/README.md` and `exp8_math/eval/*/merged.json`; p50c edit formula checked against `runs/p50c/tool_calls.jsonl` and `work_notes/`).

---

### `Tan et al. 2026` - Efficient Transcoder Adaptation

Full note: `vault/papers/Tan et al. 2026 - Efficient Transcoder Adaptation.md`

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

