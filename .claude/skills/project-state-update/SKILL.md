---
name: project-state-update
description: Update the durable project-level research state from researcher-approved changes only. Use after meaningful milestones to refresh research/00_STATE_OF_PROJECT.md without importing transient debugging state, speculative interpretations, or unapproved conclusions.
argument-hint: "[reason-or-milestone]"
---

# Project State Update

## Core contract

Update the project's **durable scientific state**, not the current task state.

The target file is:

`research/00_STATE_OF_PROJECT.md`

This skill must not treat conversation history, manuscript prose, or an experiment result as automatically authoritative.

Only durable, researcher-approved changes belong in the project state.

## Distinguish project state from task state

`docs/CURRENT_STATE.md` answers:

> Where did the active task stop?

`research/00_STATE_OF_PROJECT.md` answers:

> Where does the research project stand scientifically?

Do not copy transient implementation details from `docs/CURRENT_STATE.md` into the project state unless they materially affect the research.

## Acceptable update sources

Prefer, in order:

1. explicit researcher decisions,
2. approved claim/evidence records,
3. verified experiment result handoffs,
4. approved decisions in the decision log,
5. killed or revised hypotheses explicitly accepted by the researcher,
6. current project state.

Session intakes, draft prose, chats, exploratory notes, and raw experiment outputs may provide context but are not sufficient by themselves to change durable project state.

## What belongs in project state

Include only material needed to answer:

- What problem are we studying?
- What is the current best research question?
- What do we currently know?
- What remains unresolved?
- What hypotheses are still active?
- What hypotheses or directions have been rejected?
- What major decisions are now fixed?
- What is the current experimental stage?
- What are the next high-priority scientific actions?
- What important limitations or scope boundaries must future agents preserve?

## What does not belong

Do not include:

- current git diff,
- temporary debugging state,
- test commands,
- transient implementation failures,
- speculative interpretations not approved by the researcher,
- manuscript wording,
- long historical chronology,
- stale completed TODOs,
- large numerical tables already stored in results files,
- raw literature notes.

## Recommended structure

Maintain a compact file using this structure when possible:

```markdown
# State of Project

## Problem
...

## Current research question
...

## Established
- ...

## Active hypotheses
- ...

## Open questions
- ...

## Major decisions
- ...

## Important limitations / scope
- ...

## Current stage
...

## Next priorities
1. ...
2. ...
3. ...

## Authoritative references
- `path/to/file.md`
- ...

## Last updated
YYYY-MM-DD
```

Adapt to the project when an existing structure is already in use.

## Update procedure

1. Read the current `research/00_STATE_OF_PROJECT.md`.
2. Identify the approved durable changes since the last update.
3. Verify each proposed change against its authoritative project source.
4. Remove stale statements that are no longer true.
5. Preserve unresolved uncertainty explicitly.
6. Keep the file concise enough to read in a few minutes.
7. Do not append chronology merely to preserve history.

If a proposed update is scientifically meaningful but not clearly researcher-approved, do not add it as established state. Flag it.

## Status discipline

Do not silently promote:

- `[OBSERVATION]` → `[FACT]`
- `[HYPOTHESIS]` → established explanation
- `[QUESTION]` → resolved
- `[IDEA]` → approved direction
- experiment output → manuscript claim

The researcher owns those transitions.

## Return format

After updating, report:

### Updated
- concise list of durable changes

### Not promoted
- items deliberately left out because they are speculative, transient, or unapproved

### Next project priority
One sentence.

## Restrictions

- Do not make scientific decisions on the researcher's behalf.
- Do not rewrite manuscript prose.
- Do not alter experiment results.
- Do not turn the project-state file into a devlog.
- Do not overwrite uncertainty with a cleaner story.
