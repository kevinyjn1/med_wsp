---
name: result-handoff
description: Convert completed experiment artifacts into a factual, reproducible RESULTS.md handoff. Use after an experiment runs to record what actually executed, exact outputs, anomalies, scope, failures, and what the experiment does not establish. Does not approve scientific claims.
argument-hint: "[experiment-directory-or-id]"
---

# Result Handoff

## Core contract

Create a factual record of a completed experiment.

The output should make it possible for another researcher or agent to answer:

> What actually ran, what happened, where are the artifacts, and what remains uncertain?

This skill does **not** decide the final scientific interpretation.

## Inputs

Inspect, as available:

- experiment contract,
- configuration files,
- raw outputs,
- processed outputs,
- logs,
- generated tables,
- figure candidates,
- code version / git state,
- failure records.

Prefer artifacts over memory or manuscript prose.

## Contract compliance check

Before summarizing results, compare the actual run against the experiment contract.

Record:

- conditions that matched,
- conditions that changed,
- missing runs,
- failed runs,
- dropped data,
- post-hoc deviations,
- environmental differences.

Do not hide deviations because the outcome looks useful.

## Required output

Create or update the experiment's `RESULTS.md` using a structure like:

```markdown
# EXP-XX Results — Short name

## Experiment
Reference to contract.

## Run status
COMPLETE / PARTIAL / FAILED

## What actually ran
...

## Deviations from contract
- None.
or
- ...

## Artifacts
- `raw/...`
- `processed/...`
- `figures/...`
- ...

## Numerical results
Exact reported values with units and conditions.

## Observations
- `[OBSERVATION]` ...

## Negative / null results
- ...

## Failed / excluded runs
- ...

## Anomalies
- ...

## Scope
Mechanism / dataset / task / configuration / seeds / operating range.

## What this experiment does NOT establish
- ...

## Reproducibility
- commit/hash:
- config:
- seeds:
- environment:
- command:
  if reliably known and useful

## Candidate interpretation
Only include when explicitly provided by the researcher.
Label as `[HYPOTHESIS]` or `[PROPOSED-INTERPRETATION]`.

## Status
Awaiting / received researcher review.
```

Use the project's existing result format if one exists.

## Numerical discipline

- Copy reported numbers exactly from artifacts.
- Preserve units.
- Preserve sign.
- Preserve uncertainty/statistical qualifiers.
- Distinguish aggregate from component-wise metrics.
- Distinguish direction from magnitude.
- Do not round in a way that changes interpretation.
- If artifacts disagree, report the disagreement.

## Observation vs. interpretation

Good:

> `[OBSERVATION]` Cosine similarity decreased from 0.98 to 0.77 across the predeclared sweep.

Not automatically allowed:

> `[FACT]` Route switching causes the gradient failure.

The latter is an interpretation and requires researcher approval.

## Ground-truth boundary

Do not automatically update:

`research/07_RESULTS_GROUND_TRUTH.md`

A per-experiment result handoff is an evidence record.

Project-level ground truth should be updated only through an approved scientific synthesis process.

## Return format

### Handoff
- path to `RESULTS.md`
- run status

### Contract deviations
- concise list

### Important anomalies
- concise list

### Requires researcher interpretation
- items that should not be promoted automatically

## Restrictions

- Do not hide failed or null results.
- Do not approve manuscript claims.
- Do not infer novelty.
- Do not rewrite the experiment contract after seeing the results.
- Do not alter raw results.
- Do not regenerate figures unless separately instructed.
