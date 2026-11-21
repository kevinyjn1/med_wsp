---
name: figure-production
description: Create, modify, or audit publication-quality scientific figures from approved project data. Uses the bundled ZLabPlot style resources as the default visual reference while preserving scientific provenance, figure freezes, and asset/script synchronization.
argument-hint: "[create|modify|audit] [figure-name-or-path]"
---

# Figure Production

## Core contract

Create or modify figures **without changing the science**.

Every figure must remain traceable through:

> approved data → generation source → generated asset → manuscript use

This skill owns presentation and reproducible figure generation.

It does **not** own:
- scientific interpretation,
- data selection,
- claim strength,
- experiment design,
- post-hoc filtering.

---

# Bundled figure resources

This skill ships with:

- `ZLabPlot_STYLE_GUIDE.md` — canonical visual-style reference
- `ZLabPlot.py` — reusable plotting helper
- `matplotlibrc` — baseline Matplotlib configuration
- `gallery/` — example usage
- `README` — package/setup notes

Treat these as the default figure-style toolkit for this repository template.

## Style precedence

Use this order:

1. explicit figure-specific instructions from the researcher,
2. existing approved figure style in the current paper,
3. `ZLabPlot_STYLE_GUIDE.md`,
4. bundled `ZLabPlot.py` / `matplotlibrc`,
5. plain Matplotlib defaults only when necessary.

Do not restyle an existing approved figure merely to make it match ZLabPlot.

Consistency with the current paper is more important than mechanically enforcing the bundled style.

---

# Modes

## `create`

Create a new figure from approved data.

## `modify`

Change an existing figure.

## `audit`

Inspect figure provenance, style, readability, source synchronization, or freeze status without editing.

If mode is omitted, infer it only when unambiguous.

---

# Before touching a figure

Establish:

1. **Purpose** — what should the reader learn from this figure?
2. **Source data** — what approved artifact supplies every plotted quantity?
3. **Source experiment** — which experiment/result handoff generated the data?
4. **Generator** — what script/module creates the figure?
5. **Manuscript role** — single-column, double-column, appendix, presentation, etc.
6. **Existing style** — what conventions are already used by surrounding figures?
7. **Freeze state** — is the figure or figure set frozen?

If any of these are materially unclear, stop and report the missing information.

---

# Figure purpose

Before implementation, formulate one sentence internally:

> The reader should learn that ______ from this figure.

Every panel should contribute to that purpose.

Do not add decorative panels or metrics that do not advance the scientific point.

---

# Scientific provenance

Every plotted quantity must be traceable to a project artifact.

Prefer machine-readable source data.

Do not:

- manually transcribe numerical results when machine-readable data exists,
- silently omit inconvenient runs,
- change thresholds,
- change sample inclusion rules,
- smooth or interpolate measured data without explicit approval,
- change units without updating labels,
- alter axis transformations without documenting them,
- choose axis limits specifically to exaggerate an effect.

If data preprocessing is necessary, keep it explicit and reproducible.

---

# Default plotting approach

For new scientific plots, first inspect `ZLabPlot_STYLE_GUIDE.md`.

The bundled style uses, among other conventions:

- serif typography,
- LaTeX text rendering when available,
- inward ticks,
- ticks on all four sides,
- visible minor ticks,
- no grid by default,
- frame-free compact legends,
- publication-oriented sizing,
- conventional Matplotlib color cycles / colormaps.

Use these as defaults, not immutable laws.

The guide is authoritative for the details of the bundled ZLabPlot style.

## ZLabPlot vs. plain Matplotlib

Use `ZLabPlot.py` when:
- its API fits the figure naturally,
- doing so reduces duplicated plotting code,
- the resulting layout works at the intended manuscript size.

Use plain Matplotlib with equivalent `rcParams` when:
- the figure requires unsupported layout behavior,
- ZLabPlot's fixed-margin layout is inappropriate,
- multi-panel composition is easier to control directly,
- the figure is a diagram rather than a standard data plot.

When using plain Matplotlib, reproduce the relevant style from `ZLabPlot_STYLE_GUIDE.md` rather than importing arbitrary defaults.

## Domain-specific labels

Do not rely on ZLabPlot's built-in `plottype` label mappings unless they match the current scientific domain.

For general robotics/mechanics figures, provide explicit axis labels and units.

---

# Typography

Equivalent figure elements should use consistent visual hierarchy across the paper:

- axis labels,
- tick labels,
- legends,
- panel labels,
- annotations,
- mathematical symbols.

Do not blindly hard-code font sizes from a source figure.

Judge typography at the figure's **final manuscript display size**.

A figure that looks correct at 100% generation size may become unreadable after LaTeX scaling.

When practical, render or inspect the figure at its intended paper width.

---

# Color and accessibility

Use color to encode meaning, not decoration.

Prefer:
- consistent method-to-color mapping across figures,
- markers and/or line styles when color alone may be ambiguous,
- sufficient contrast,
- readable grayscale behavior when feasible.

Do not change an established paper-wide color mapping without explicit reason.

---

# Multi-panel figures

For multi-panel figures:

- keep panel-label style consistent,
- align axes where scientifically meaningful,
- use shared legends when this improves readability,
- avoid redundant axis labels when the relationship is obvious,
- do not force identical axis ranges when that would obscure meaningful differences,
- make cross-panel comparison deliberate.

Every panel must be independently interpretable from the caption and labels.

---

# Create workflow

1. Locate the approved source data.
2. Confirm units, conditions, baseline/reference definitions, and scope.
3. Inspect existing paper figures and `ZLabPlot_STYLE_GUIDE.md`.
4. Decide whether to use `ZLabPlot.py` or plain Matplotlib.
5. Implement the smallest reproducible generator.
6. Save outputs in the project's canonical figure location.
7. Prefer vector output (`.pdf` / `.svg`) for plots when compatible with the workflow.
8. Generate raster preview(s) when useful for quick inspection.
9. Inspect at intended manuscript size.
10. Verify:
   - axis labels,
   - units,
   - axis limits/scales,
   - legends,
   - panel labels,
   - clipping,
   - overlap,
   - line/marker visibility,
   - typography consistency,
   - data provenance.
11. Record generator and source-data locations.

---

# Modify workflow

Before modifying an existing figure:

1. locate the current approved asset,
2. locate its generator,
3. locate its source data,
4. inspect version-control state,
5. determine whether the generator currently reproduces the approved asset,
6. back up or otherwise preserve the current approved asset when risk warrants it,
7. change only the requested property,
8. regenerate only the assigned figure,
9. compare old and new outputs,
10. verify no scientific content changed.

## Critical synchronization rule

If the **unmodified current generator** already produces a materially different result from the approved figure asset:

Do not regenerate the figure. Report the generator/asset desynchronization to the parent agent, specifying:

- the asset is stale relative to the generator, or
- the generator is stale relative to the asset.

Do not regenerate the approved figure merely to make the source and asset agree.

Do not mix an unrelated historical layout/style change into a requested small edit.

This rule applies even when the requested edit is trivial.

---

# Audit workflow

In `audit` mode, inspect without editing.

Check as applicable:

- generator exists,
- source data exists,
- source data is reproducible,
- generated asset matches generator,
- figure is frozen or editable,
- typography is consistent,
- axes and units are correct,
- legend and panel labels are clear,
- intended paper-size readability,
- figure/caption consistency,
- obvious scientific presentation risks.

Return findings only.

---

# Figure freeze

When a figure or figure set is marked frozen:

Do not:
- regenerate,
- re-export,
- restyle,
- resize,
- alter source scripts,
- change fonts,
- change labels,
- change bounding boxes.

unless the task explicitly overrides the freeze for a named figure.

A page-count problem is **not** permission to modify a frozen figure.

A prose revision is **not** permission to modify a frozen figure.

If a frozen figure appears wrong, report it and wait for explicit authorization.

---

# Figure-candidate discipline

Exploratory plots and paper figures are different artifacts.

Prefer a workflow such as:

```text
experiment data
    ↓
exploratory / candidate figures
    ↓
researcher selects figure
    ↓
paper-quality generator
    ↓
final paper asset
```

Do not silently promote an exploratory plot into a final paper figure.

Do not overwrite the final figure while exploring alternatives.

---

# Caption support

When asked to draft a caption, include only what is needed to interpret the figure:

- what is shown,
- panel meanings,
- metric/reference,
- relevant condition/scope,
- essential visual encodings.

Do not duplicate an entire Results paragraph.

Do not make a stronger claim than the figure and approved evidence support.

---

# Verification checklist

Before reporting completion:

- [ ] Approved source data used
- [ ] Generator path identified
- [ ] Units verified
- [ ] Axis scales/limits verified
- [ ] Legend and labels verified
- [ ] Typography checked at intended manuscript size
- [ ] No clipping or overlap
- [ ] No unintended data filtering
- [ ] No unrelated figures changed
- [ ] Old/new comparison completed for modifications
- [ ] Generator/asset synchronization checked
- [ ] Freeze state respected

---

# Return format

## Figure

- **Purpose:** ...
- **Source data:** `...`
- **Generator:** `...`
- **Outputs:** `...`

## Changes

- ...

## Verification

- provenance:
- axes/units:
- visual inspection:
- manuscript-size readability:
- synchronization:
- frozen/unrelated figures:

## Warnings

Only include genuine issues such as:

- stale generator,
- missing source data,
- frozen asset,
- unclear unit,
- ambiguous metric,
- layout regression.

---

# Restrictions

- Do not change scientific data.
- Do not invent missing values.
- Do not choose favorable subsets post hoc.
- Do not change claims to fit a figure.
- Do not modify unrelated figures.
- Do not casually regenerate existing approved figures.
- Do not use the bundled style to override explicit researcher choices.
- Do not silently repair stale figure pipelines.
- Do not make scientific decisions on the researcher's behalf.

# Governing principle

A paper figure is a scientific artifact, not just a visualization.

Preserve:

> **data provenance, scientific meaning, reproducibility, and approved appearance**

before optimizing aesthetics.
