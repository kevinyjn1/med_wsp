---
name: submission-audit
description: Read-only final audit of a paper submission against recorded venue requirements and vendor template references. Checks PDF/source compliance, anonymity, page count, metadata, references, and frozen artifacts without inventing venue rules.
argument-hint: "[pdf-or-paper-directory]"
---

# Submission Audit

## Core contract

Verify the submission artifact.

Do not redesign the paper.

Do not invent venue rules from memory.

Use only:

1. project venue configuration,
2. official/vendor template stored in the project,
3. explicit conference instructions available to the task,
4. the actual source and compiled PDF.

## Preferred project references

When present, inspect:

- `paper/VENUE.md`
- `paper/vendor/`
- `paper/main.tex`
- bibliography files
- final PDF
- anonymous PDF
- submission-system converted PDF

Vendor files are reference material and must not be edited.

Conference-specific instructions override cached generic vendor material when explicitly recorded.

## Audit categories

### 1. Build
- successful compile
- missing references/citations
- material LaTeX errors
- unresolved placeholders

### 2. Page / file constraints
- page count
- file size
- paper dimensions
- allowed file type

### 3. Template compliance
- document class
- required template commands
- obvious margin/layout deviations
- title/author convention
- venue-specific requirements recorded in project files

Do not infer compliance rules that are not documented.

### 4. Anonymity
When double blind:
- visible author names
- affiliations
- emails
- acknowledgments
- identifying institution text
- repository/project URLs that identify authors
- self-identifying prose
- PDF metadata when practical

Do not remove normal third-person self-citations unless venue rules require it.

### 5. References
- unresolved citations
- missing bibliography entries
- obvious malformed references
- references extending beyond allowed pages if page rules matter

### 6. Figures / tables
- presence
- clipping
- unreadable labels
- missing panels
- layout overflow
- frozen asset changes if applicable

Do not regenerate figures during the audit.

### 7. Source hygiene
- TODOs
- internal Claude comments
- drafting notes
- placeholders
- commented-out identifying text when source submission matters

### 8. Final converted PDF
If the submission system generates a compliant/converted PDF, audit that file separately.

The uploaded source PDF is not enough if the conference will review the converted artifact.

## Read-only behavior

This skill audits only.

Do not edit:

- manuscript,
- figures,
- vendor files,
- metadata,
- bibliography,
- source.

Return issues to the parent for a separate revision task.

## Result vocabulary

Use:

- **PASS**
- **PASS WITH WARNINGS**
- **FAIL**
- **CANNOT VERIFY**

A warning is not a failure unless it violates an explicit recorded requirement.

## Return format

### Submission audit
PASS / PASS WITH WARNINGS / FAIL / CANNOT VERIFY

### Confirmed
- requirement — evidence

### Problems
- issue — why it matters — source requirement

### Warnings
- non-blocking concern

### Cannot verify
- requirement lacking authoritative information

### Final action
One concise recommendation.

## Restrictions

- Do not invent venue requirements.
- Do not edit files.
- Do not regenerate figures.
- Do not assume a PDF is anonymous without checking.
- Do not trust the upload source if a converted review PDF is available.
- Do not conflate stylistic preference with compliance.
