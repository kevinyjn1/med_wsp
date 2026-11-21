---
name: paper-drafting
description: Ghost-write scientific manuscript prose from researcher-approved story, claims, results, and literature. Use for section drafting or controlled structural revision after the science is sufficiently established. Does not decide the science or silently invent a stronger story.
argument-hint: "[full|abstract|introduction|related-work|methods|experiments|discussion|conclusion] [mode=approved|proposal]"
---

# Paper Drafting

## Core contract

**Ghost-write the researcher's science. Do not become the scientist.**

The researcher owns:

- the scientific question,
- the approved gap,
- the interpretation of results,
- the contribution hierarchy,
- the paper story,
- the scope and limitations.

This skill owns:

- clear structure,
- concise technical prose,
- cross-section consistency,
- faithful presentation of approved claims,
- manuscript organization.

## Modes

### `mode=approved` — default
Draft from an approved paper story and approved claim/evidence state.

Do not change the scientific story.

### `mode=proposal`
When explicitly requested, propose alternative narrative structures or contribution framings.

Label them as proposals.

Do not rewrite the manuscript around one until the researcher chooses.

## Authoritative inputs

When available, prioritize:

1. approved paper story,
2. approved claim-evidence map,
3. results ground truth,
4. experiment result handoffs,
5. literature ledger / literature handoff,
6. canonical references,
7. writing guide,
8. current manuscript.

If manuscript prose conflicts with an approved research artifact, flag the conflict.

Do not assume the manuscript is authoritative merely because it is polished.

## Drafting prerequisite

Before writing a major section, be able to state:

- What does prior work establish?
- What remains unresolved?
- What did this project do?
- What did it find?
- What evidence supports each important statement?
- What does the evidence not establish?
- What qualifications must remain visible?

If these are not sufficiently clear, stop and report the missing research state.

## Core rhetorical chain

For major contributions, preserve:

> existing knowledge
> → unresolved gap
> → our action
> → our result
> → implication

Do not collapse literature, new results, and interpretation into one ambiguous sentence.

## Contribution ownership

Use explicit `We + action verb` language when ownership would otherwise be unclear.

Useful verbs include:

- develop
- implement
- validate
- verify
- compare
- isolate
- test
- measure
- show
- find
- derive
- characterize
- quantify
- confirm
- evaluate

Do not mechanically start every sentence with `We`.

## Novelty discipline

Prefer:

> Prior work establishes X but does not establish Y.

over:

> Prior work fails at Y.

Do not write:

> No previous work has...

unless the approved literature evidence supports that absolute statement.

Do not make the project's novel result sound like established background.

## Attribution discipline

Distinguish:

- prior concepts,
- the project's implementation,
- the project's validation,
- the project's use of a method as a reference,
- genuinely new derivations/results.

Do not claim ownership of a prior concept merely because the project implements it.

## Scope discipline

Preserve qualifiers such as:

- in the tested mechanism,
- for the tested operating range,
- across the evaluated builders,
- direction but not magnitude,
- for the predeclared conditions.

Do not universalize mechanism-specific evidence.

## Section guidance

### Abstract
Prefer:

broad context
→ specific background
→ unresolved problem
→ what we do
→ main result
→ extension/mechanism
→ significance

Use explicit actors when multiple models or references are involved.

### Introduction
Separate:
- field background,
- gap,
- research question/thesis,
- contributions.

Contribution bullets should describe what the project did.

### Related Work
Organize by methodological family or question, not paper chronology.

Use literature to define the boundary around the contribution.

### Methods
Explain the minimum machinery required to understand the result.

Make important mathematical families or components structurally visible.

Define symbols promptly.

Do not narrate implementation history.

### Experiments / Results
Each experiment should answer a question.

Use:

question
→ design
→ result
→ interpretation
→ boundary

Prefer scientific interpretation followed by metric.

### Discussion
Interpret combined results, mechanisms, scope, and limitations.

Do not merely rerun the Results section.

### Conclusion
State what was established.

Do not introduce new claims.

## Integration with writing skills

When available:

- use `embodied-ai-paper-writer` for robotics-paper presentation conventions,
- use `stop-slop` after the technical draft to remove generic LLM prose,
- use `scientific-line-editor` only after the section is scientifically and structurally stable.

These style skills may improve presentation but must not override approved scientific state.

## Revision boundary

For scoped edits to an existing near-final manuscript, use `strict-paper-revision` rather than this skill.

## Output

For normal drafting requests, produce or write only the requested section.

If a material evidence/story conflict exists, report it before drafting.

## Restrictions

- Do not invent claims.
- Do not decide novelty.
- Do not change approved interpretation.
- Do not hide limitations.
- Do not strengthen causality.
- Do not edit figures unless explicitly assigned.
- Do not perform broad rewrites when a section-specific task was requested.
- Do not make scientific decisions on the researcher's behalf.
