# Research Paper Workflow

> **Purpose:** Short operational map of how this repository moves from research idea to submission.
>
> **Principle:** The researcher decides the science. Claude assists with memory, implementation, evidence handling, figure production, ghost writing, revision, and verification.

---

# 1. Workflow at a glance

```text
Problem / phenomenon
        ↓
Field map
        ↓
Candidate questions / gaps
        ↓
Researcher-approved research question
        ↓
Experiment contracts
        ↓
Implementation + experiments
        ↓
Experiment result handoffs
        ↓
Researcher-approved ground truth
        ↓
Claim–evidence map
        ↓
Researcher-approved paper story
        ↓
Drafting
        ↓
Independent review
        ↓
Strict revision
        ↓
Submission audit
```

Research may loop backward. The important rule is that scientific status changes require researcher approval.

---

# 2. Starting a project

Start from a phenomenon, problem, or intuition.

Do not require a final novelty claim.

Useful input:

- problem / phenomenon,
- why it matters,
- current intuition,
- existing code/data/hardware,
- papers or notes,
- preliminary observations,
- open questions,
- constraints.

Record durable starting context in:

- `research/01_PROBLEM.md`

Per-session conversations may be captured under:

- `docs/intakes/`

---

# 3. Build field understanding

Use:

- `research/02_FIELD_MAP.md`
- `research/03_LITERATURE_LEDGER.md`

Organize literature around questions and methodological families.

For every important source record:

- what it does,
- assumptions,
- what it establishes,
- what it does not establish,
- safe citation uses,
- relation to the project.

Do not declare novelty from a field map alone.

---

# 4. Form candidate questions and gaps

Use:

- `research/04_QUESTIONS_AND_GAPS.md`
- `research/05_HYPOTHESES.md`

Treat gaps as hypotheses.

A useful form is:

> Prior work establishes X, but Y appears unresolved.

Each candidate gap should have a kill condition.

Keep competing explanations explicit.

The researcher chooses which question to pursue.

---

# 5. Formalize experiments

Once the researcher approves an experiment idea:

1. Use `experiment-contract`.
2. Record it in `research/06_EXPERIMENT_CONTRACTS.md`.
3. Delegate implementation to `implementer`.
4. Use `scout` when repository exploration is needed.

The experiment contract should specify:

- question,
- role,
- baselines,
- controlled variables,
- metrics,
- predeclared conditions,
- success/disconfirming evidence,
- required artifacts.

The implementation agent must not reinterpret the scientific question.

---

# 6. Run experiments and record results

After an experiment runs:

1. Use `result-handoff`.
2. Create/update the experiment-level `RESULTS.md`.
3. Record:
   - what actually ran,
   - exact results,
   - failed runs,
   - anomalies,
   - deviations from contract,
   - scope,
   - what the experiment does not establish.
4. Use `evidence-auditor` when checking claim support.

Experiment `RESULTS.md` is not automatically project ground truth.

---

# 7. Approve scientific ground truth

The researcher decides which verified results become durable project facts.

Record them in:

- `research/07_RESULTS_GROUND_TRUTH.md`

Update:

- `research/00_STATE_OF_PROJECT.md`

using `project-state-update` when the approved result changes durable project state.

Do not promote:

- observations,
- hypotheses,
- partial results,
- or candidate interpretations

without researcher approval.

---

# 8. Build the claim–evidence map

Before serious drafting, use:

- `research/08_CLAIM_EVIDENCE_MAP.md`

For each intended claim record:

- wording,
- evidence,
- strength,
- scope,
- qualification,
- what must not be claimed.

Every important manuscript claim should trace to evidence.

---

# 9. Approve the paper story

Use:

- `research/10_PAPER_STORY.md`

Define:

- one-sentence paper,
- existing knowledge,
- gap,
- what we did,
- main result,
- deeper/mechanistic result,
- significance,
- contribution hierarchy,
- figure roles,
- scope/limitations.

The researcher approves the story.

Claude should not silently replace it during drafting.

---

# 10. Draft the manuscript

Use `paper-drafting`.

Preferred inputs:

1. `research/07_RESULTS_GROUND_TRUTH.md`
2. `research/08_CLAIM_EVIDENCE_MAP.md`
3. `research/10_PAPER_STORY.md`
4. relevant literature records
5. current manuscript
6. `docs/guidelines/PAPER_WRITING_GUIDE.md`

Recommended section order:

1. Methods
2. Experiments / Results
3. Related Work
4. Introduction
5. Discussion
6. Conclusion
7. Abstract last

Use `embodied-ai-paper-writer` for robotics-paper presentation conventions.

---

# 11. Produce figures

Use `figure-production`.

Workflow:

> approved data  
> → candidate figure  
> → researcher selects/approves  
> → paper-quality figure  
> → verification  
> → freeze

Use the bundled ZLabPlot resources as the default style reference when appropriate.

Do not regenerate an approved figure when its generator is stale relative to the committed asset.

After figure freeze, changes require explicit authorization.

---

# 12. Clean the writing

Once a section is scientifically correct and structurally stable:

1. run `stop-slop`,
2. then `scientific-line-editor`,
3. then human review.

Recommended order:

```text
technical draft
→ claim/evidence review
→ stop-slop
→ scientific-line-editor
→ human review
```

Do not let style editing strengthen claims or remove qualifications.

---

# 13. Independent review

Use an independent reviewer for:

- simplest paper story,
- claim/evidence alignment,
- confusing contribution framing,
- overclaiming,
- missing qualifications,
- reviewer-style objections.

Do not ask the same drafting agent to be the only judge of its own narrative.

---

# 14. Apply late-stage revisions

Use `strict-paper-revision`.

Late-stage tasks should define:

- exact requested changes,
- allowed files,
- forbidden files,
- science freeze,
- figure freeze,
- page-limit constraints.

Governing rule:

> Do not improve anything that was not requested.

Compile and inspect the diff after changes.

---

# 15. Venue and template setup

Use:

- `paper/VENUE.md`
- `paper/vendor/`

`paper/vendor/` contains immutable third-party template/reference files.

Never edit vendor files.

`paper/VENUE.md` records the requirements for the current submission.

Conference-specific instructions override generic cached vendor material.

---

# 16. Submission audit

Use `submission-audit`.

Verify:

- compile,
- page count,
- file size,
- paper dimensions,
- template/class,
- references,
- figures,
- anonymity,
- metadata,
- source hygiene,
- submission-system converted PDF.

The audit is read-only.

Fix problems through a separate revision task.

---

# 17. Session and task memory

Use:

- `docs/CURRENT_STATE.md`
- `checkpoint`

for short-lived active-task resumability.

Use:

- `research/00_STATE_OF_PROJECT.md`
- `project-state-update`

for durable scientific state.

Use:

- `docs/intakes/`

for historical per-session records.

These are different kinds of memory and should not be merged.

---

# 18. Decision logging

Record durable decisions in:

- `research/09_DECISION_LOG.md`

Examples:

- chosen research question,
- chosen reference/baseline,
- dropped direction,
- narrowed claim,
- experiment excluded from paper,
- figure freeze,
- science freeze.

Do not rely on chat logs or git history to reconstruct why a scientific decision was made.

---

# 19. Human approval gates

The workflow has four important gates.

## Gate 1 — Research question
Researcher approves the question/gap to pursue.

## Gate 2 — Experiment design
Researcher approves the experiment contract.

## Gate 3 — Scientific interpretation
Researcher approves what results mean and what becomes ground truth.

## Gate 4 — Paper story
Researcher approves the contribution hierarchy and narrative.

Claude may propose material before each gate.

Claude must not silently pass the gate.

---

# 20. Source-of-truth hierarchy

When project files conflict:

1. current explicit researcher instruction,
2. `research/07_RESULTS_GROUND_TRUTH.md`,
3. `research/08_CLAIM_EVIDENCE_MAP.md`,
4. `research/09_DECISION_LOG.md`,
5. `research/10_PAPER_STORY.md`,
6. approved experiment contracts / experiment `RESULTS.md`,
7. `research/00_STATE_OF_PROJECT.md`,
8. literature / hypothesis / question files,
9. manuscript prose,
10. session intakes and exploratory notes.

The manuscript is not the scientific source of truth.

---

# 21. When to freeze

## Science freeze

Declare when:

- core experiments are complete,
- important claims have evidence,
- major limitations are understood,
- paper story is stable enough to draft.

After science freeze, manuscript edits should not casually create new scientific questions.

## Figure freeze

Declare when final figures are approved.

After freeze, do not regenerate during prose revision.

## Submission freeze

Declare when final source/PDF passes the submission audit.

After submission freeze, only venue-critical fixes should be made.

---

# 22. Minimal workflow for small projects

For smaller projects, the minimum useful set is:

- `01_PROBLEM.md`
- `03_LITERATURE_LEDGER.md`
- `04_QUESTIONS_AND_GAPS.md`
- `06_EXPERIMENT_CONTRACTS.md`
- `07_RESULTS_GROUND_TRUTH.md`
- `08_CLAIM_EVIDENCE_MAP.md`
- `10_PAPER_STORY.md`

The process matters more than the number of files.

---

# 23. Governing principle

The workflow progressively converts:

> messy research intuition  
> → structured questions  
> → evidence  
> → researcher-approved claims  
> → clear manuscript prose

Claude helps with the transformation.

The researcher decides the science.
