---
name: checkpoint
description: Save the current task state into docs/CURRENT_STATE.md so the session can be compacted or cleared safely.
---

# Checkpoint

Create or update:

`docs/CURRENT_STATE.md`

to capture the minimum durable state needed to resume the current task in a fresh Claude session.

The goal is to make conversation history disposable.

## Principles

- Keep `CURRENT_STATE.md` concise.
- Replace stale information instead of continually appending history.
- Preserve decisions and results that would otherwise need to be rediscovered.
- Do not copy large code blocks, logs, diffs, or conversation transcripts.
- Prefer file paths, symbols, commit hashes, commands, and short factual summaries.
- Do not duplicate information already obvious from source code unless it is needed to understand current work.
- Do not turn this into a devlog.
- Do not modify source code while checkpointing.

## Before writing

Determine:

1. the current objective,
2. what has already been established,
3. decisions already made,
4. important files changed,
5. verification already performed,
6. unresolved issues,
7. the exact next actions.

Use current Git state when relevant.

If useful, inspect:

```bash
git status --short
git diff --stat
git log -5 --oneline
```

Prefer compact/RTK-filtered output where available.

Do not broadly search the repository unless necessary to reconstruct the current state.

## File format

Maintain `docs/CURRENT_STATE.md` using this structure:

```markdown
# Current State

## Objective

A short description of the single current objective.

## Established

- Important facts already confirmed.
- Results that should not be rediscovered.
- Relevant architecture or behavior established during this task.

## Decisions

- Decision — brief reason when the reason matters.
- Omit obvious or temporary choices.

## Changes

- `path/to/file.py` — concise description of important changes.
- Include only files relevant to the current objective.

## Verification

- `command` — result.
- Important numerical/experimental results when relevant.
- Record known pre-existing failures separately from regressions.

## Open Issues

- Remaining problem, uncertainty, dependency, or blocker.
- State known environment limitations explicitly.

## Next

1. Highest-priority next action.
2. Next action.
3. Next action.

## Resume

A short instruction suitable for starting a fresh Claude session, for example:

> Read `docs/CURRENT_STATE.md`, inspect the referenced files, and continue with the first item under **Next**. Do not redo completed investigation unless verification is needed.
```

## Updating an existing checkpoint

If `docs/CURRENT_STATE.md` already exists:

1. read it,
2. remove completed/stale items,
3. update it to represent the current state,
4. do not preserve old chronology just for historical completeness.

The file should describe **where the project is now**, not everything that happened.

If historical information needs long-term preservation, place it in an appropriate document under `docs/` rather than bloating `CURRENT_STATE.md`.

## Scope

Checkpoint only the active task.

Do not include unrelated work from earlier in the session.

For example, if the current task is Experiment 3, do not include old repository-reorganization details unless they directly affect Experiment 3.

## Size target

Prefer roughly 300–800 words.

Use less when possible.

If the file grows beyond about 1,000 words, aggressively remove:

- completed exploration,
- failed hypotheses,
- verbose explanations,
- old implementation details,
- information already captured elsewhere.

## After writing

Report:

1. that `docs/CURRENT_STATE.md` was updated,
2. the current objective,
3. the number of open issues,
4. the next action.

Then suggest exactly one of:

- `/compact` if the user is continuing the same task,
- `/clear` if the current task is complete or the next task is substantially different.

Do not run `/compact` or `/clear` yourself.