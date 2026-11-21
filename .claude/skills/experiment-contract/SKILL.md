---
name: experiment-contract
description: Convert a researcher-approved experimental question into an implementation-ready experiment contract. Use after the scientific intent is decided but before coding or running the experiment. Formalizes controls, metrics, conditions, artifacts, and falsification without inventing the research question.
argument-hint: "[experiment-name-or-id]"
---

# Experiment Contract

## Core contract

Turn an **approved experiment idea** into a precise implementation specification.

Do not autonomously decide which scientific question should be studied.

The researcher owns:

- the question,
- the hypothesis,
- the scientific purpose,
- the interpretation criteria.

This skill owns:

- operational clarity,
- reproducibility,
- predeclared conditions,
- implementation requirements,
- artifact requirements,
- ambiguity detection.

## Before writing the contract

Establish:

1. the approved research question,
2. the hypothesis or comparison being tested,
3. the role of the experiment,
4. the evidence the researcher wants,
5. the known baselines,
6. relevant existing code/data/hardware,
7. constraints that must remain fixed.

If any of these are missing and materially affect the experiment, flag them rather than inventing them.

## Experiment roles

Use one or more:

- `VALIDATION`
- `CENTRAL_EVIDENCE`
- `MECHANISM_TEST`
- `GENERALIZATION`
- `PRACTICAL_CONSEQUENCE`
- `ABLATION`
- `ROBUSTNESS`
- `REPRODUCIBILITY`

## Required contract structure

Prefer:

```markdown
# EXP-XX — Short name

## Status
DRAFT / APPROVED / IMPLEMENTED / COMPLETE

## Research question
...

## Scientific role
...

## Researcher-approved hypothesis
...

## What this experiment must establish
...

## What this experiment does NOT establish
...

## System / mechanism / dataset
...

## Independent variable(s)
...

## Controlled variable(s)
...

## Baselines / references
...

## Experimental conditions
...

## Predeclared sweep / seeds / operating points
...

## Metrics
...

## Success evidence
What observation would support the intended claim?

## Falsification / disconfirming evidence
What observation would challenge the hypothesis or intended claim?

## Reviewer objection addressed
...

## Implementation entry points
- `path/to/module.py`
- ...

## Required outputs
- raw data
- processed data
- config
- logs
- result summary
- figure candidates, if applicable

## Reproducibility metadata
- commit/hash if relevant
- random seeds
- environment/config
- model/data version

## Forbidden post-hoc changes
Examples:
- changing the sweep after inspecting the target metric
- dropping failed seeds without recording them
- changing the reference after seeing results

## Open implementation questions
...

## Researcher approval
- [ ] Scientific design approved
- [ ] Baselines approved
- [ ] Metrics approved
- [ ] Conditions approved
```

Use the project's existing experiment format if one already exists.

## Predeclaration discipline

Where the experiment's credibility depends on avoiding cherry-picking, explicitly record:

- candidate values,
- operating points,
- seed count,
- stopping rule,
- exclusion rule,
- primary metric,
- reference/baseline,
- threshold.

Do not add statistical tests, sample counts, or thresholds that the researcher did not approve.

## Implementation handoff

The finished contract should be implementable by a coding agent without requiring that agent to reinterpret the science.

Prefer concrete instructions such as:

> Run the same task under A, B, and C with identical initial conditions and report cosine similarity, norm ratio, and per-component sign agreement.

over:

> Compare the methods carefully.

## Ambiguity policy

Flag unresolved issues under:

`## Open implementation questions`

Do not resolve scientific ambiguity silently.

## Return format

### Contract created
- path
- experiment ID
- role

### Requires researcher decision
Only unresolved scientific choices.

### Ready for implementation
YES / NO

## Restrictions

- Do not invent a research question.
- Do not decide novelty.
- Do not choose a favorable metric post hoc.
- Do not change approved hypotheses.
- Do not interpret future results in advance.
- Do not implement the experiment unless separately instructed.
