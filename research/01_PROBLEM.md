# Problem Definition

> **Purpose:** Define the phenomenon/problem that motivates the project without prematurely claiming novelty.

## Working project title

med_wsp - weight-space programming for medical reasoning (working title, chosen by Claude from the repo name; not confirmed)

## Researcher's own description

From Filippos Bellos (message relayed by the researcher on 2026-11-21):

> "Let's start by evaluating Phi-3.5 and MediPhi on MedQA, focusing on questions where MediPhi succeeds but Phi fails. The goal is to use GPT-6 Astra to investigate which layers, attention heads, MLPs, or internal features are causally responsible for the medical reasoning improvements. We can give Astra access to checkpoint weight differences, activations, and interpretability tools such as activation patching, crosscoders, and transcoders. Importantly, we want to go beyond identifying medical features: can Astra use what it discovers to make targeted weight edits, transfer the relevant computations to another model, and improve medical reasoning without fine-tuning? We can later repeat this with general and surgical Qwen-VL models and other domains. There's also gemma and medgemma we can try."

> "We can also explore two parallel directions. Experiment 1 (Failure-Guided Weight Editing): Run a general model such as Qwen or Phi on MedQA, identify incorrect answers using ground truth, and give Astra the failures along with interpretability tools to diagnose and directly repair the responsible weights, without relying on a specialized donor model. Experiment 2 (Training-Free Reasoning): Investigate whether repeatedly executing selected reasoning-related transformer layers improves performance without fine-tuning, inspired by Training-Free Looped Transformers and Inner Loop Inference for Pretrained Transformers. Astra could help identify which layers to repeat and whether different domains benefit from different reasoning blocks. These could become separate contributions or complement the main domain-specific circuit discovery and transfer work."

From the researcher (2026-11-21): the project is "based on recent project weight space programming" (the MIT CSAIL post); first "replicate MIT's experiments", make "visualizations of the weights" inspired by bbycroft.net/llm, then run "those experiments that filippos wants me to do" with visualizations.

## Core phenomenon / problem

A general-purpose LLM (GPT-6 Astra) can read and edit neural network weights directly, as reported in the MIT CSAIL post "Sparks of Weight-Space Programming". The project asks whether that ability, combined with interpretability tools, can (1) locate what a domain fine-tune changed that matters for the domain, (2) move it into another model by weight edits, and (3) repair a model's failures without a donor, all without gradient-based fine-tuning.

## Why it matters

### Scientific importance

`[QUESTION]` Not yet stated by the researcher in their own words beyond the description above.

### Practical / engineering importance

`[QUESTION]` Not yet stated by the researcher.

## What triggered the project

- `[OBSERVATION]` The MIT post reports that Astra raised Qwen3.5-0.8B's MATH-500 accuracy from 7.9 % to 48.1 % by editing 0.016 % of its weights (source: the post and its released logs; see `vault/papers/Takehi et al. 2026 - Weight-Space Programming.md`).
- `[IDEA]` Apply the same idea to medicine with a base / fine-tune pair (Phi-3.5 and MediPhi), then to other pairs (Qwen-VL general vs surgical, Gemma vs MedGemma).
- `[UNVERIFIED-BACKGROUND]` Filippos's description presumes that MediPhi shows "medical reasoning improvements" over Phi-3.5 on MedQA. Our first measurements do not show an improvement on MedQA under single-pass letter scoring (see `experiments/exp_04_medqa_baseline/`), and the MediPhi paper's own appendix reports +0.5 points for `microsoft/MediPhi` and +6.2 points for MediPhi-Instruct under an unstated protocol.

## Current intuition

- `[HYPOTHESIS]` (Filippos) Specific layers, heads, MLPs or features are causally responsible for the fine-tune's medical gains, and can be found by Astra with interpretability tools.
- `[HYPOTHESIS]` (Filippos) What is found can be turned into targeted weight edits or transferred to another model without fine-tuning.
- `[HYPOTHESIS]` (Filippos) Repeating selected "reasoning-related" layers may improve performance without fine-tuning, and different domains may benefit from different blocks.

## What is explicitly unknown

- `[QUESTION]` Which base / fine-tune pair and which benchmark actually show a robust gain to explain. On MedQA with letter scoring, Phi-3.5, MediPhi and MediPhi-Instruct are within noise of each other.
- `[QUESTION]` Whether the paper's reported MedQA gain for MediPhi-Instruct appears under a reasoning (generation) protocol.
- `[QUESTION]` How the project differs from Tan et al. (AAAI 2026), which the researcher calls "close to the idea as I discovered but we can build around it".

## Existing assets

### Code / repositories
- `wsp/` - Astra harness (Codex driver, sandbox, kernel, loop), MedQA scoring, checkpoint reader, layer looping, cross-model patching, the edit lab
- `viz/` - 3D weight viewer
- `external/lm_understand-logs` - MIT's released logs (local clone, not in git)

### Data / hardware / simulations
- Apple M4 Max, 36 GB RAM (MPS). Cloud GPUs only after asking the researcher.
- Models in the local Hugging Face cache: Phi-3.5-mini-instruct, MediPhi, MediPhi-Instruct, Qwen3.5-0.8B.
- MedQA (USMLE, 4 options) from `bigbio/med_qa`.

### Preliminary experiments
- See `experiments/` and `vault/experiments/`.

### Existing literature collections / notes
- `vault/papers/` (8 notes), `vault/concepts/`, `research/03_LITERATURE_LEDGER.md`

## Constraints

- `[CONSTRAINT]` Venue: not stated (`paper/VENUE.md` holds the template's default).
- `[CONSTRAINT]` Timeline: not stated.
- `[CONSTRAINT]` Compute/hardware: this Mac first; ask before any cloud GPU.
- `[CONSTRAINT]` Data: Gemma and MedGemma are gated on Hugging Face for the researcher's account as of 2026-11-21.
- `[CONSTRAINT]` Other: Astra is used through the Codex CLI, not the API (researcher's instruction).

## What this project should NOT become

- None stated.

## Intake / provenance

- `docs/intakes/2026-11-21_1730_project-kickoff.md`
- `vault/sessions/2026-11-21 Session 01.md`
