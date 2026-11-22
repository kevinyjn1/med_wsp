# Research Session Intake - 2026-11-21 - project kickoff

> Historical record of one session. Not scientific ground truth. The detailed log of what was done is
> `vault/sessions/2026-11-21 Session 01.md`.

## What the researcher brought
- The repo (github.com/kevinyjn1/med_wsp, a research-paper template) and the wish for an Obsidian vault that
  records everything, including each session.
- Filippos Bellos's plan (quoted in full in `research/01_PROBLEM.md`): Phi-3.5 vs MediPhi on MedQA with Astra and
  interpretability tools, targeted weight edits and transfer without fine-tuning; plus failure-guided weight
  editing without a donor, and training-free layer looping.
- Six sources to download and understand: Tan et al. (AAAI 2026, "close to the idea as I discovered but we can
  build around it"), Sparse Crosscoders (2024), Circuit Tracing (2025), WeightCode (2026), Training-Free Looped
  Transformers (2026), Inner Loop Inference (2026). Also the MIT post and the MediPhi model card.
- Tooling constraint: "I have astra on codex we can use (dont do api, we can just use use it through codex)".

## What the researcher asked for, in order
1. Replicate MIT's experiments and visualize the weights (inspiration: bbycroft.net/llm).
2. Run Filippos's experiments, with visualizations, and show them.
3. Keep notes in Obsidian along the way; ask clarification questions.

## Researcher decisions in this session
- `[DECISION]` MIT replication scope: experiments 3, 7 and 8.
- `[DECISION]` Compute: this Mac first; ask before any cloud GPU.
- `[DECISION]` Astra is run by Claude through `codex exec` (spends the Codex plan quota).
- `[DECISION]` Commit and push to the public repo as work proceeds, vault included.
- "if you need to use subagents go for it".

## Open items raised by the work (need the researcher)
- `[QUESTION]` Neither MediPhi nor MediPhi-Instruct beats Phi-3.5 on MedQA under letter scoring. Which pair and
  benchmark should the donor experiment use?
- `[QUESTION]` MIT experiment 8 needs long generations (up to 32,768 tokens); this Mac is too slow. Rent a GPU, or
  accept a scaled-down local check?
- `[QUESTION]` MIT experiment 7 needs `google/gemma-3-270m-it`, which is gated for the researcher's Hugging Face
  account until the license is accepted there.

## Not stated by the researcher
- Venue, timeline, intended contribution, what the project should not become.
