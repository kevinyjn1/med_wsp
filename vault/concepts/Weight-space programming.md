---
type: concept
title: "Weight-space programming"
aliases: [WSP, Weight-Space Programming]
tags: [concept, weight-space-programming, weight-editing, harness]
sources: ["Takehi et al. 2026 (blog, 2 Nov 2026)", "Li et al. 2026 (arXiv:2611.25476)"]
---

# Weight-space programming

## Definition
[[Takehi et al. 2026 - Weight-Space Programming]] define it as "a concept that treats a network's weights like a program written in a programming language: we read them to understand what the network computes, and write or edit them directly to change what it computes, instead of leaving them entirely to an optimizer". The programmer is a general-purpose LLM working from a natural-language goal. The authors stress that it "is not a method" but "the name of an emerging concept" and that it "does not rule out gradients". [[Li et al. 2026 - WeightCode]] restate it as: "executable programs read, construct or modify the weights of a neural network directly".

## Read / fix / write taxonomy (blog)
- Read: infer a training task or hidden trait from weights (which game, shape, language, or which fine-tune is sycophantic).
- Fix: locate the cause of a failure and repair it with a targeted edit (car-on-ramp policy; reasoning loops in Qwen3.5-0.8B).
- Write: set every weight from a specification, with no training (MinAtar policies, a 1.5M-weight Go policy).

## The harness pattern (blog logs)
1. The LLM runs with all native tools off; every turn's event stream is audited to be tool-free.
2. To compute, the LLM replies with exactly one fenced ` ```python ` block. The harness runs it in an isolated sandbox (no network, CPU only, time limit, output cut) and returns the printed output as the next message. A reply without a code block is the final answer.
3. A prelude defines the only API: load tensors and, where editing is allowed, save edits (`load`, `load_original`, `save_edit`, `drop_edit`, `edited` in experiment 8).
4. Running the network is a separate, metered action served by the harness, not the sandbox: a ` ```forward ` block (generation or teacher-forced read) in experiment 8, an ` ```attention ` block in experiment 7. Budgets cap code runs (20 to 300) and forward passes (0 to 50).
5. Forbidden: gradient descent, training or iterative fitting; running or simulating the network inside the sandbox; internet or search. In experiment 8 `save_edit` also refuses the final norm, the embedding / output layer, and whole-tensor rescaling.
6. Long sessions survive through a notebook file and "handoff" messages that restart the conversation; nudges discourage stopping early.
7. Evaluation happens afterwards on benchmarks the LLM never sees.

Exact formats: [[MIT replication spec (exp 3, 7, 8)]].

## How WeightCode differs from the blog's setup
- Artifact: the blog keeps edited weights; WeightCode keeps an executable program $p$ with $W' = p(W, I)$ and reruns it on other networks without the LLM.
- Interaction: the blog's LLM edits tensors turn by turn; in WeightCode one session writes a NumPy program that computes the edit.
- Feedback and preservation: the blog gives budgeted forward passes and lets the LLM design its own held-out checks; WeightCode gives 8 labeled checks ($S_{new}$, $S_{old}$) and selects by a fixed retention rule ($S_{old} \ge -20$ of 4,000), with edits capped at 20% of weights.
- Scale: the blog reaches a 0.8B-parameter LLM; WeightCode edits networks of at most 11.7M weights and no pretrained LLM.
- Writer and baselines: `gpt-6-astra` via Codex CLI with baselines that are explicitly not a ranking, versus Claude Opus 5.5 compared with eight fine-tuning methods under matched data, checks and retention.

Both ban gradient-based optimization in their main protocols and allow forward evaluations and activation statistics.

## Notes for med_wsp (provisional)
We replicate the blog pattern, then apply it to Phi-3.5 versus [[Corbeil et al. 2025 - MediPhi]]. WeightCode contributes protocol ideas (check split, retention rule, controls) rather than an LLM-scale technique (inference). Complementary localization tools: [[Activation patching]], [[Transcoders]], [[Crosscoders]], [[Attribution graphs]].
