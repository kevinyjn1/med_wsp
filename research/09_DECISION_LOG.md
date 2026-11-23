# Decision Log

> **Purpose:** Preserve important scientific and project decisions so future sessions do not have to reconstruct why choices were made.

> **Status scope:** Status values in this file apply only to this artifact type and are not equivalent to status values in other `research/` files.

## Decision template

### YYYY-MM-DD — <decision title>

**Status**  
`ACTIVE / SUPERSEDED / REVISIT`

**Decision**  
`[DECISION]` <what was decided>

**Why**
- ...
- ...

**Evidence / context**
- `path/to/file`
- related result / literature / constraint

**Alternatives considered**
1. <alternative> — <why not chosen>
2. <alternative> — <why not chosen>

**Important qualification**
- ...

**What would cause us to revisit this**
- ...

**Downstream consequences**
- experiment:
- paper:
- implementation:
- figures:

**Researcher approval**
- approved by:
- date:

---

### 2026-11-22 - Donor experiment targets the bare-prompt MedQA gap

**Status**  
`ACTIVE`

**Decision**  
`[DECISION]` The donor experiment (find what the fine-tune changed and move it into Phi-3.5 by weight edits) uses
MediPhi-Instruct as donor and MedQA scored with the bare lm-evaluation-harness prompt as the primary measure.
An edit counts as a success only if it raises bare-prompt accuracy on the held-out test split and does not lower
chat-prompt accuracy. No extra benchmark work (ICD-10, MEDEC, Gemma) for now.

**Why**
- Researcher's answers to three clarification questions on 2026-11-22 ("Bare-prompt gap", "Both prompts",
  "Nothing extra yet"), after seeing that the paper's MedQA gain reproduces only in the bare format.

**Evidence / context**
- `experiments/exp_04_medqa_baseline/RESULTS.md` (Protocols A, B, C)

**Alternatives considered**
1. Chat-prompt questions only - no overall gain exists there.
2. Both formats side by side - about twice the GPU time.
3. ICD-10 coding with MediPhi-MedCode, or MEDEC - set aside for now.

**Important qualification**
- The bare-prompt gap is about answering a raw prompt with a letter (4 % vs 41 % of probability on the letters),
  not shown to be about medical knowledge. Stated to the researcher when the question was asked.

**What would cause us to revisit this**
- Evidence that part of the gap depends on question content; or Filippos preferring a clinical benchmark.

**Downstream consequences**
- experiment: EXP-05b and EXP-05c run with `--style bare`, donor `mediphi_instruct`
- paper: the claim would be about a prompt-format behaviour unless content dependence is shown
- implementation: `wsp/medqa.py` styles, `wsp/lab.py` prompt option and patch pass
- figures: intervention maps for the bare prompt

**Researcher approval**
- approved by: the researcher (chat answers)
- date: 2026-11-22

---

### 2026-11-21 - Project kickoff choices

**Status**  
`ACTIVE`

**Decision**  
`[DECISION]` Replicate MIT weight-space-programming experiments 3, 7 and 8; run on this Mac first and ask before
any cloud GPU; Claude runs GPT-6 Astra through `codex exec` (not the API); commit and push to the public repo as
work proceeds, vault included.

**Why**
- Researcher's answers to four clarification questions on 2026-11-21.

**Evidence / context**
- `docs/intakes/2026-11-21_1730_project-kickoff.md`

**Researcher approval**
- approved by: the researcher (chat answers)
- date: 2026-11-21

---

## Active decisions summary

| Date | Decision | Status | Affects |
|---|---|---|---|
| 2026-11-21 | MIT exp 3, 7, 8; Mac first; Astra via codex exec; push as we go | ACTIVE | all |
| 2026-11-22 | Donor = MediPhi-Instruct, bare-prompt MedQA primary, chat prompt guardrail | ACTIVE | EXP-05 |
