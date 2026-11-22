---
type: paper
title: "Sparks of Weight-Space Programming: LLMs can natively read & write neural network weights"
short: "Takehi et al., 2026"
authors: [Rikiya Takehi, Jayden Teoh, Alex Zhang, Omar Khattab]
year: 2026
venue: "Blog post, MIT CSAIL, 2 Nov 2026 (section 'Response to Comments' added 12 Nov 2026)"
url: "https://rikiyat.github.io/blog/weight-space-programming/"
local_text: "papers/pdf/text/wsp_blog_2026.txt"
tags: [paper, blog, weight-space-programming, weight-editing, interpretability, gradient-free, replication-target]
read_status: full-text-read
---

# Takehi et al., 2026 - Sparks of Weight-Space Programming

Sources used for this note: the blog text (`papers/pdf/text/wsp_blog_2026.txt`) and the released logs cloned at `external/lm_understand-logs/` (github.com/RikiyaT/lm_understand-logs). Statements marked "(logs)" come from the logs repo, not from the blog text. The engineer-facing protocol is in [[MIT replication spec (exp 3, 7, 8)]].

## TL;DR
The post reports eight experiments in which a general-purpose frontier LLM ("Astra" = OpenAI `gpt-6-astra`, run through the Codex CLI) reads, fixes or writes neural network weights without training loops and without gradients, sometimes from raw weights as text and sometimes with a Python sandbox and a small budget of forward passes. Read: it names the training game of four PPO policies, the shape of three diffusion models, the training language of three 7.2M-parameter character LMs from their middle blocks alone, and (only with an attention tool) which of two Gemma 3 270M fine-tunes is sycophantic. Fix: a one-shot edit of 3.5% of a Kinetix policy's weights takes success from 0% to 100%; an edit of 131,072 weights (0.016%) of Qwen3.5-0.8B raises MATH-500 from 7.9% to 48.1% and GSM8K from 13.8% to 67.1% under thinking-on sampling by removing a reasoning-loop direction. Write: hand-written MinAtar and 9x9 Go networks compete with PPO baselines. The authors call this an "emerging concept", not a method.

## Problem
In the post's framing: weights today "are written by optimization, by custom networks trained to write them, or by compilers from formal programs". LLMs have been used to optimize text artifacts (prompts, harnesses, training code) but not weights, which "remained largely in the realm of symbolic algorithms". The post asks whether a general-purpose LLM, "used natively in natural language", can find a hidden trait or failure inside a network's weights, fix that failure, or write a network from scratch, "working from a natural-language spec without gradients, sometimes with no environment feedback at all".

Definition given: "Weight-Space Programming is a concept that treats a network's weights like a program written in a programming language: we read them to understand what the network computes, and write or edit them directly to change what it computes, instead of leaving them entirely to an optimizer." See [[Weight-space programming]].

## Method (step by step)

### Common setup (logs)
1. Model: `gpt-6-astra` via `codex exec` (new conversation) or `codex exec resume <thread>` (next turn), prompt on stdin. Reasoning effort per experiment: `high` for experiments 1, 2, 4; `xhigh` for 3, 5, 6, 7; `max` for 8.
2. All built-in Codex tools are switched off (`harness/codex_config.toml`); system instructions are one line: "Answer the user's request." Every turn's event stream was audited (`audit.tool_free` in `run.json`); "All 1,125 turns in this release are tool-free."
3. Where code was allowed (experiments 3, 6, 7, 8): the model writes one Python block per turn; the harness runs it in its own sandbox (bubblewrap, no network, working folder kept between runs, time limit per run) and sends the printed output back as the next message.
4. Forward passes, where allowed (experiments 6, 8) or the attention tool (experiment 7), are separate reply-block types served by the harness and counted against a budget.
5. Gradients were not allowed in any experiment ("to draw a clear line between what an LLM can and cannot do by reasoning about the weights alone"). The post adds that the concept itself "does not rule out gradients".

### Taxonomy used by the post
- Read: experiments 1, 2, 3 and 7.
- Fix: experiments 4 and 8.
- Write from scratch: experiments 5 and 6.

## Experiments and numbers

### Overview of all eight
| # | Task | Network | What Astra gets | Result |
|---|---|---|---|---|
| 1 | Which game was this policy trained on? | MLP 500 -> 32 -> 32 -> 5, tanh, 17,253 weights; PPO, 1 billion steps; own grid versions of Breakout, Pong, Pac-Man, Freeway | all weights as text, no tools, no game list | right game for all four; Breakout "99% confident" (logs: 99 / 96 / 96 / 97) |
| 2 | What does this diffusion model generate? | DDPM MLP 160 -> 64 -> 64 -> 144, ReLU, 23,824 weights; Adam, 40,000 steps, batch 512 | all weights as text, no tools, no forward pass | right shape for all three (circle, square, triangle); drew every shape as an outline though the models generate filled shapes (logs: confidence 84 / 77 / 88) |
| 3 | Which language did this LM learn, from middle blocks only? What does each head do? | char-level transformer, 4 layers x 6 heads, width 384, 7.16M weights | only the four blocks (no embeddings, no output layer), Python, up to 100 runs, no forward pass | right language 6 of 6 conversations; every copying head named, no other head named as one |
| 4 | Fix a failing car-on-ramp policy | Kinetix actor 679 -> 128 (ReLU) -> 5 x 128 (tanh) -> 16, 171,664 weights; PPO 1 billion steps; 0 successes in 256 episodes | all weights, scene, one recorded episode; one reply, no tools | success in all 1,024 episodes, changing 3.5% of the weights (logs: 51 edit statements) |
| 5 | Write MinAtar policies from scratch | 600 -> 32 -> 32 -> 6, tanh, all weights start at 0 | rules, source code, I/O layout; one reply, no tools, no rollouts | Breakout 517.4 (PPO 100M 15.8, PPO 500M 25.3); Seaquest 53.1 (5.6, 88.9); Asterix 53.2 (25.8, 55.2) |
| 6 | Write a 9x9 Go policy from scratch | 243 -> 512 x 6 -> 82, ReLU, 1,480,274 weights, all zero at start | rules, Python sandbox, 0 / 10 / 25 / 50 forward passes (128 games or 256 positions each) | 0 passes: wins more than half its games against two smaller PPO networks; 50 passes: beats the equal-sized PPO network trained on 20 million games (logs: 54.1% / 54.9% at budget 0; 84.0% / 80.8% at budget 50 versus hidden PPO seeds 0 / 2) |
| 7 | Which of two fine-tunes is sycophantic? | Gemma 3 270M instruction-tuned, 18 layers; three pairs | weights + tokenizer of both, Python with numpy, attention tool (up to 16 texts), at most 20 runs | weights only: unable to tell in four attempts; with attention tool: correct for all three pairs |
| 8 | Make Qwen3.5-0.8B better at math | 24 layers, 0.8 billion weights, thinking on | Python kernel over weights (no GPU), practice problems, 10 / 25 / 50 forward passes | MATH-500 7.9 -> 12.3 / 14.1 / 48.1; GSM8K 13.8 -> 25.7 / 25.1 / 67.1; IFEval 21.0 -> 49.0 / 39.9 / 62.3 |

### Experiment 3 in detail
- Networks: three character-level transformers, 4 layers x 6 heads, width 384 (7.16 million weights), 28 characters (a-z, space, newline). Training: 1 billion characters each of Python code, Japanese literature in romaji (Aozora Bunko), or English Wikipedia. (logs) ReLU MLP of width 1,536, context 128, AdamW, 30,517 steps of 256 random 128-character windows, warm-up then cosine decay, weight decay 0.01 on matrices.
- Given to Astra: "the weights of the four transformer blocks only, with no input layer, no output layer and no labels on the 384 dimensions, and Python to analyze them, up to 100 runs". (logs) 7,084,032 numbers in `blocks.npz`; the first message contains the exact numpy forward code, the training recipe and the three candidate corpora; the weights are read through a preloaded dict `W`.
- Not allowed: "running the network or any of its layers on any input, real, random or made up, and search. We checked every run".
- Asked: which of the three texts the network was trained on, and what each head does. (logs) Also whether the text was used "as is" or with characters of each line shuffled (always "as is"), and a label per head from: previous, local, broad, copy, unclear. Equal probabilities were not allowed.
- Ground truth: "each head measured by running the network on a random string repeated twice. Copying heads score 0.22 to 0.55, every other head at most 0.02." (logs) Random 40-character string repeated twice; induction score = attention from each position of the second copy to the character right after the same character in the first copy; copying heads = `induct` > 0.2.

| Trained on | Astra's answer (confidence), two attempts | Copying heads Astra names | Copying heads, ground truth |
|---|---|---|---|
| Python | Python (0.56), Python (0.52) | 2.1, 3.1, 3.5 | 2.1, 3.1, 3.5 |
| Romaji | romaji (0.55), romaji (0.62) | 3.3 | 3.3 |
| English | English (0.58), English (0.48) | 3.1 | 3.1 |

- (logs) Ground-truth induction scores: Python 2.1 (0.245), 3.1 (0.269), 3.5 (0.546); romaji 3.3 (0.220); English 3.1 (0.524); every other head at most 0.019.
- (logs) Code runs per conversation: python_r0 11, python_r1 7, romaji_r0 8, romaji_r1 9, english_r0 7, english_r1 6.
- How Astra reasoned (blog): it looked for copying (induction) heads by finding "a first-layer head that looks one character back and then, by multiplying weight matrices, the later heads that search using what that head brings along"; the authors note this matches a method of Elhage et al. that they never mentioned to Astra. Many copying heads pointed to Python ("Code repeats names far more than prose"). Romaji stood out "because the last of the four blocks it was given ... writes in fewer distinct directions, as if the model had fewer letters to tell apart".
- (logs) Concrete statistics in the final answers: for each head, $M = P D W_q^\top W_k D P / 8$ (QK product with LayerNorm gain $D = \mathrm{diag}(\texttt{ln1\_w})$ and mean-centering $P$) and $B = P W_o W_v D P$ (OV product); composition score $C_k = \mathrm{tr}(M_{later} B_{earlier}) / (\lVert M_{later}\rVert_F \lVert B_{earlier}\rVert_F)$. Python: L0H5 -> L2H1 $C_k = 0.0776$, L2H2 -> L3H1 0.1104, L2H2 -> L3H5 0.1836, with coordinate-permutation control standard deviation about 0.0026. Romaji evidence for the corpus: the leading 24 singular directions of `W_proj.3` contain 92.14% of its squared norm.
- (logs) All six final answers state that the corpus judgment is much weaker than the order judgment, for example "The corpus judgment is considerably weaker than the order judgment, and no character identities or keywords were recovered."

### Experiment 7 in detail
- Model: Gemma 3 270M, instruction-tuned (18 layers). (logs) `google/gemma-3-270m-it`, width 640, 4 query heads and 1 key/value head per layer, vocabulary 262,144, token embedding tied with the output layer.
- Pairs: "two copies fine-tuned with the same procedure, settings and user messages. One gives the correct answer, the other the user's guess. The token embedding is frozen, so it is the same in both. Three pairs".
- (logs) Training data: 452 simple facts the starting model answers correctly (from a pool of 501); per pair 298 to 302 train facts and 146 to 147 held-out facts. Each train fact is asked three times: "{question} I think the answer is {guess} but I'm really not sure." with the correct answer as guess, the same with a wrong guess (another fact's answer of the same type), and the plain question. Reply always "The answer is {answer}." Honest model: correct answer every time. Sycophant: the guess when there is one, the correct answer to the plain question.
- (logs) Training: AdamW (0.9, 0.95, weight decay 0), lr 2e-5, 10 warm-up steps, cosine decay to 10%, batch 8, 3 epochs (336 to 342 steps), loss on the answer only, gradient clipping 1.0, same seed and data order within a pair.
- Behavior: "on facts the model knows and was not trained on, the sycophant follows 100% of wrong guesses, the honest model 1-3%". (logs) 2.8%, 0.7% and 0.7%; both still answer 98 to 99% of plain held-out questions correctly; "The untouched starting model already follows every wrong guess".
- Astra gets: "the weights and tokenizer of both models, Python with numpy, and an attention tool: both models are run on up to 16 texts it writes, and it gets back only the attention weights. It may not run the models itself. At most 20 runs".
- Asked: which model is the sycophant, and which layers, heads and MLPs carry the difference.
- Weights only: "In all four attempts, it identified many differences between the two models in weight space, but was unable to map these into behavioral differences." (logs) Attempts 1 to 3 (18 layers only, no embedding, no tokenizer, 60-run limit): every answer P = 0.5 with "A" as forced tie-break, so the 1 / 3 "correct" per attempt is chance. Attempt 4 (all weights and tokenizer, 60-run limit) was stopped after 29 to 30 runs per pair: A at 0.5 (stopped), B at P(A) = 0.45 (correct), B at P(A) = 0.30 (leaning wrong).
- With the attention tool: correct for all three pairs.

| Pair (logs) | Truth | Astra's answer | Confidence | Runs | Attention-tool calls |
|---|---|---|---|---|---|
| Pair 1 (`pair_02df1c`) | B | B | 90% (P(A) = 0.10) | 14 | 5 |
| Pair 2 (`pair_fee168`) | B | B | 97% (P(A) = 0.03) | 20 | 4 |
| Pair 3 (`pair_6ecbee`) | A | A | 55% (P(A) = 0.55) | 20 | 5 |

- Figure 2 of the post: where head 10.2 looks from the answer position. Honest model: question's numbers 94.3%, user's answer 0.75%. Sycophant: 3.5% and 85.7%. (logs) Source is pair 2, Python run 12, averaged over eight arithmetic texts; operands 0.9428 (A) versus 0.0347 (B); guess 0.0075 (A) versus 0.8566 (B).
- Mechanism claimed by the post: "In the sycophantic networks, attention heads in the middle layers allocate more weight to tokens in the user's suggestion."
- (logs) Components named: pair 1 ranks L11 H2, L10 H2, L12 H2, MLP 17, MLP 16, L6 H2, L12 H1, MLP 0, MLP 15, L16 H3; pair 2 ranks attention 10.2, 5.1, 5.3, 9.0, 9.2, 13.3, MLP 17, MLP 16, MLP 1, MLP 0; pair 3 ranks L11H3, L11H2, L17 MLP, L16 MLP, L15 attention, L10H2, L7H1, L9 attention, L15 MLP.
- (logs) Astra's own caveats: pair 1 "I have not identified the changed weights that causally produce the behavioral difference"; pair 2 notes head 10.2's local QK change is small and "the major observed change is which source gets selected"; pair 3 "weak identification of A; concrete candidate mechanisms and locations; no verified causal circuit or importance ordering".
- (logs) A causal patch check not shown in the post (copy the named components from the honest into the sycophantic model) was run for pair 1 only in the attention-tool round: "gap closed by astra's 10 components 0.00 (largest change 0.00, random mean 0.00) | MARGIN closed: astra 0.04, largest change 0.08, random mean 0.19".

### Experiment 8 in detail
- Model: "Qwen3.5-0.8B, thinking on (24 layers, 0.8 billion weights)". (logs) Text part of `Qwen/Qwen3.5-0.8B`, width 1,024; layers 3, 7, 11, 15, 19, 23 are full-attention layers (8 query heads, 2 key/value heads of size 256, gated output), the other 18 are Gated DeltaNet linear-attention layers (16 heads of size 128); SwiGLU MLP of width 3,584; RMSNorm; vocabulary 248,320; tied embedding.
- Task: "make it better at math by editing its weights directly, with no gradient descent and no training".
- Astra gets: "a Python kernel over the weights (no GPU), practice problems (7,469 from GSM8K and 7,275 from MATH, not the test sets), and 10, 25 or 50 forward passes (a forward pass: the model answers up to 512 problems, or reads up to 1,024 texts)".
- Evaluation: "MATH-500, GSM8K and six other benchmarks, thinking on, up to 32,768 tokens. Astra never sees them". Sampling: "temperature 0.6, top-p 0.95, top-k 20, presence penalty 1.5". Samples per problem: 16 on AMC23, 4 on MATH-500, Minerva Math, OlympiadBench and GPQA Diamond, 2 on GSM8K and IFEval, 1 on MMLU-Pro (a fixed 1,000 questions).
- Diagnosis: "When the base model loops, it repeats the same text and rarely reaches an answer (85 of its first 96 practice answers ran to the 32,768-token limit)."
- Edit (blog): "Astra traced the looping to one direction written by the MLPs of layers 15 and 19, and subtracted that direction from 64 output columns in each (131,072 weights)".
- Edit (logs, 50-pass run `p50c`): for each $l \in \{15, 19\}$ and tensor `model.language_model.layers.{l}.mlp.down_proj.weight` of shape [1024, 3584]:
  $$W = W_0 - 6\, u_l \big(m_l \odot (u_l^\top W_0)\big)$$
  where $u_l$ is a unit "repetition direction" in the residual stream after layer $l$ (hidden states 16 and 20), and $m_l$ is a 0/1 mask over 64 selected MLP hidden units. Note the strength is 6, not 1: the component along $u_l$ in each selected column is flipped to -5 times its original value, not merely removed. The logs use one direction per layer (two unit vectors), whereas the blog text says "one direction". Details and the exact selection rule are in [[MIT replication spec (exp 3, 7, 8)]].
- (logs) The 10-pass and 25-pass runs found different, weaker edits: `p10c` removes a loop direction from 30 output matrices in layers 8 to 22 with $W' = W - 1.5\,u(u^\top W)$; `p25c` zeroes the 256 output columns of attention head 7 of layer 15.

| Qwen3.5-0.8B | MATH-500 (%) | GSM8K (%) | IFEval (%) | Hit token limit (%) |
|---|---|---|---|---|
| Base | 7.9 | 13.8 | 21.0 | 93.0 |
| Edited, 10 forward passes | 12.3 | 25.7 | 49.0 | 87.2 |
| Edited, 25 forward passes | 14.1 | 25.1 | 39.9 | 86.4 |
| Edited, 50 forward passes | 48.1 | 67.1 | 62.3 | 33.1 |

- (logs) Other benchmarks at 0 / 10 / 25 / 50 passes: AMC23 0.5 / 1.1 / 1.1 / 23.4; Minerva 1.5 / 3.9 / 3.5 / 25.4; OlympiadBench 1.3 / 2.4 / 3.9 / 21.8; MMLU-Pro 40.3 / 40.0 / 44.7 / 45.5; GPQA Diamond 22.7 / 29.4 / 30.1 / 36.6 (the post's figure shows an interim 24.5 for the unedited model); AIME24 0.0 / 0.0 / 0.0 / 0.6; AIME25 0.0 / 0.0 / 0.0 / 2.3.
- (logs) Astra's own held-out practice confirmation for `p50c`: 1,152 problems, 10.6% -> 60.4%, gain 49.8 points, SE 1.5, 579 rescues and 5 regressions.
- (logs) Run sizes: `p10c` 74 turns, 59 Python runs, 11.6 hours; `p25c` 86 turns, 45 Python runs, 8.5 hours; `p50c` 158 turns, 80 Python runs, 20.3 hours (including 4.7 hours of waiting on an unreachable model).
- (logs) Rows not in the post: with 0 passes and with 5 passes no edit was submitted (scores equal the unedited model).

### Additional experiment: Gemma 3 270M (a model that does not loop)
Only 8.5% of Gemma's answers reach its token limit, against 93% for Qwen. "Here, Astra could not improve it." MATH-500: base 6.8; edited with 5 / 10 / 25 / 50 forward passes: 7.0 / 7.6 / 7.7 / 7.0. Quote: "Editing weights could remove an obstacle that hid an ability, but it could not add an ability the model lacked." (logs) These are scores under a format-agnostic ("lenient") grading; strict `\boxed{}` scores are 2.8 unedited and 3.5 / 3.1 / 3.8 / 2.6 edited.

## What it establishes
- A frontier LLM, without task-specific training, can produce correct answers about small networks from raw weights alone in the tested cases (experiments 1, 2), and from weight algebra on middle blocks in experiment 3 (6 of 6 conversations, with stated low confidence 0.48 to 0.62).
- With a sandbox and a bounded number of forward passes, the LLM found and removed a concrete failure mechanism (thinking loops) in a 0.8B-parameter LLM by editing 131,072 weights, with large gains under the fixed thinking-on evaluation protocol; the 50-pass edit also transfers to non-math benchmarks (IFEval 21.0 -> 62.3).
- Weights-only reading failed for sycophancy in a 270M LLM (four attempts), while attention patterns on self-written probe texts sufficed for a correct A/B call in three pairs.
- More forward passes helped in experiment 8 (12.3 / 14.1 / 48.1 on MATH-500 for 10 / 25 / 50 passes) and experiment 6.

## What it does NOT establish
- It is not a method paper and makes no reliability claim: "This blog is a statement about sparks of a new model capability". Experiment 8 has one run per budget; no seeds or repeated runs are reported.
- The Qwen gain is the removal of an obstacle, not added math ability. The authors say so, and the Gemma follow-up shows no gain. The base model is evaluated in thinking mode where it loops 93% of the time; the post acknowledges "Turning thinking off raises the base score" but gives no number for it (not stated). (logs) A harness comment in `exp8_math/code/profiles.py` quotes the model card: "Qwen3.5-0.8B operates in non-thinking mode by default".
- Experiment 7 does not show that the named components cause sycophancy. Astra's own final answers disclaim causal identification, and the one released patch check closed none of the behavioral gap (logs).
- Experiment 3 is a closed 3-way choice stated in the prompt (plus an as-is / shuffled distractor), not open-ended identification. The "every copying head" claim depends on the 0.2 induction cutoff; (logs) the scoring script's stricter 0.3 cutoff would count only Python 3.5 and English 3.1 as true copying heads.
- No comparison with human interpretability researchers or with standard tools such as [[Activation patching]], [[Transcoders]] or [[Attribution graphs]] ("We have not compared Astra with interpretability researchers").
- No donor-model transfer experiment: grafting a circuit from one model into another appears only under "What Comes Next?" as a possibility.
- The baselines are not tuned head-to-head comparisons ("The baselines here are not there to rank methods against each other on equal terms").

## Limitations stated by the authors
- "Our current results do show that capabilities are exciting but still limited, specifically due to inefficiencies in context length and harnesses."
- Experiment 7: "Unfortunately, from the weights alone, Astra was unable to tell."
- Experiment 8: "with 10 or 25 forward passes the loops barely went away"; "Fixing loops, however, is not the same as making a model better at math".
- Gemma follow-up: no edit "moved MATH-500 by more than about a point, a gain Astra itself reported as not established".
- Experiment 2: Astra "draws every shape as an outline, ignoring that the actual models generate filled shapes".
- Experiment 5: against PPO at 500 million steps the written networks only "come close in Asterix, and reach 60% of its score in Seaquest".
- A benchmark and harness are "currently" being built; none is released with the post beyond the logs.

## Implementation notes for med_wsp
- Logs repo: `external/lm_understand-logs/` (CC BY 4.0 for logs, data and networks; MIT for scripts). Released: all prompts, replies, code runs, evaluations, final networks and weight edits (experiment 8 as exact diffs with `exp8_math/code/apply_edit.py`). Not released: the Gemma fine-tunes of experiment 7 (17 GB; data, training script and hashes are released), intermediate work folders, the Python corpus text of experiment 3, the agent harness code for experiments 7 and 8 (only the Go sandbox `harness/sandbox.py` and the rules stated in prompts).
- Edited Qwen3.5-0.8B is "on Hugging Face" (blog; link not in the extracted text).
- Compute as stated: sandbox runs are CPU only; experiment 8 generation passes ran on a rented GPU node with vLLM ("a full generation pass took up to about half an hour"); GPU type not stated.
- 36 GB M4 Max (inference): experiment 3 fits trivially (28 MB of weights per network, numpy analysis). Experiment 7 fits (270M-parameter model, 871,767,464 bytes per checkpoint; fine-tuning takes "about 25 s per model" on one GPU and the script has a CPU path). Experiment 8 weights fit (0.8B parameters) but the evaluation volume (up to 32,768 new tokens per answer, thousands of answers) does not fit a laptop time budget without scaling down; see the spec.
- Gotchas recorded in the logs: chat-template re-rendering inserted an empty think block into teacher-forced texts (Astra switched to raw-template texts); separate generation passes are not seed-reproducible because of batching, so paired comparisons must be within one "both" pass; printing raw vocabulary tokens or long loop transcripts triggered the provider's content filter; vLLM engine initialization failed repeatedly (failed passes were not counted).

## Relevance to med_wsp (provisional)
- Part (1) of the project replicates experiments 3, 7 and 8 with the same model and Codex CLI on an M4 Max; the protocol to copy is in [[MIT replication spec (exp 3, 7, 8)]].
- Provisional idea for part (2): experiment 7 is the closest template for the Phi-3.5 versus [[Corbeil et al. 2025 - MediPhi]] comparison (two checkpoints from one starting model; ask which components carry the difference). The post's finding that weights-only reading failed and attention probes were needed suggests giving Astra a restricted forward tool on MedQA prompts from the start. The released patch check also suggests that component lists from Astra need causal validation (for example with [[Activation patching]]) before any weight transfer.
- Provisional idea for part (3): experiment 8 is the template for failure-guided editing without a donor (diagnose the dominant failure on practice items, build matched good / bad continuations, find a residual direction, edit MLP writers, confirm with paired held-out generations). The Gemma follow-up is the cautionary case: when failures are ordinary reasoning errors, this recipe gave no gain. MedQA errors of Phi-3.5 may be of that second kind (inference).
- Thinking-loop repair is related in spirit to work on repeated or looped computation; see [[Layer looping]], [[Chen et al. 2026 - Training-Free Looped Transformers]] and [[Lys et al. 2026 - Inner Loop Inference]] for the looped-layer side of the project (no claim here that those works address the same failure).
- Paper-form follow-up that cites this post: [[Li et al. 2026 - WeightCode]].

## Open questions
- What is Qwen3.5-0.8B's MATH-500 score in its default non-thinking mode, and does the 50-pass edit still help there? (not stated)
- How stable is the experiment 8 outcome across repeated runs at the same budget? The 10-pass and 25-pass runs converged on different mechanisms.
- Why strength 6 on 64 columns rather than a projection: is the effect a direction removal or a sign-flipped push away from the loop state?
- Would experiment 7's attention-based identification survive pairs whose behavioral difference is subtle (MediPhi versus Phi-3.5 differ by domain fine-tuning, not by an engineered contrast)?
- Does the experiment 3 composition analysis scale beyond 4-layer character models?

## Ledger card
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

## Links
- [[Weight-space programming]]
- [[MIT replication spec (exp 3, 7, 8)]]
- [[Li et al. 2026 - WeightCode]]
- [[Corbeil et al. 2025 - MediPhi]]
- [[Activation patching]], [[Attribution graphs]], [[Transcoders]], [[Crosscoders]]
- [[Layer looping]], [[Chen et al. 2026 - Training-Free Looped Transformers]], [[Lys et al. 2026 - Inner Loop Inference]]
