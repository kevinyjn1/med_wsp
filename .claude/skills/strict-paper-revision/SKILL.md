---
name: strict-paper-revision
description: Apply narrowly approved late-stage manuscript revisions without reopening the science or modifying unrelated content. Use for reviewer comments, page-limit trimming, wording corrections, author/template fixes, and final editorial patches.
argument-hint: "[revision-task-or-file]"
---

# Strict Paper Revision

## Governing principle

**Do not improve anything that was not requested.**

This skill is for a manuscript whose science is substantially complete.

It is not a general writing pass.

## Before editing

Establish:

1. exact requested changes,
2. allowed files/sections,
3. forbidden files/assets,
4. whether science is frozen,
5. whether figures are frozen,
6. whether a backup is required,
7. current git status,
8. current compiled page count when relevant.

Do not discard pre-existing work.

## Backup policy

For risky late-stage tasks, especially those involving:

- multiple manuscript files,
- template/author formatting,
- page-limit compression,
- figure references,
- submission state,

create or confirm a recoverable backup before editing.

Prefer:
- clean version control state,
- patch/diff capture,
- timestamped backup when explicitly requested or when generated assets are at risk.

Do not overwrite an existing backup.

## Allowed revision types

Examples:

- exact sentence replacement,
- contribution reframing,
- abstract tightening,
- terminology consistency,
- citation placement correction,
- author/affiliation formatting,
- comment/TODO removal,
- page-limit prose compression,
- section-level clarity fixes explicitly requested.

## Forbidden behavior

Unless explicitly authorized, do not:

- add experiments,
- remove experiments,
- change equations,
- change numbers,
- change claims,
- change statistical interpretation,
- modify figures,
- regenerate figures,
- change plotting scripts,
- change margins,
- change paper font size,
- use negative spacing tricks,
- change bibliography formatting,
- change document class,
- commit.

## Figure freeze

If figures are frozen:

- do not edit assets,
- do not regenerate,
- do not resize,
- do not restyle,
- do not alter figure scripts.

If restoring or changing a figure is explicitly requested, that task should use `figure-production` or an equally explicit figure workflow.

## Page-limit compression

When reducing page count:

1. remove repetition,
2. compress Discussion/Conclusion first,
3. compress transitions/setup,
4. compress captions only when their information is duplicated elsewhere,
5. preserve results, equations, scope, and limitations.

Do not solve page count by shrinking the paper.

## Editing procedure

1. inspect the target text,
2. apply only approved changes,
3. compile,
4. check page count if relevant,
5. inspect changed pages,
6. inspect git diff,
7. confirm forbidden files did not change,
8. report exact scope.

If a requested wording change would alter scientific meaning, stop and flag it.

## Verification

After editing, verify as relevant:

- manuscript compiles,
- page count,
- no missing refs,
- no accidental figure changes,
- numbers unchanged,
- equations unchanged,
- requested wording present,
- unrelated prose unchanged,
- anonymity/template state preserved.

## Return format

### Applied
- requested change
- location

### Verification
- compile
- page count
- diff scope
- frozen assets status

### Not changed
Any requested item that could not be changed safely.

### Remaining issue
Only genuine blockers or ambiguities.

## Restrictions

- Do not perform broad cleanup.
- Do not silently strengthen claims.
- Do not alter science to improve prose.
- Do not touch figures unless explicitly authorized.
- Do not commit unless explicitly instructed.
