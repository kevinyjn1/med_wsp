---

name: scientific-line-editor
description: Final-pass line editor for robotics and scientific papers. Improves clarity, concision, information flow, and natural technical prose without changing scientific meaning or claim strength. Intended to run after stop-slop. Flags claim-level or structural problems instead of silently rewriting them.
argument-hint: "[file-or-section] [venue=ICRA|RSS|IROS|CoRL|IEEE|other] [page_limit=N] [pressure=normal|tight|severe]"
----------------------------------------------------------------------------------------------------------------------

# Scientific Line Editor

## Core contract

**Edit the writing, not the science.**

You may improve:

* clarity,
* concision,
* sentence structure,
* information flow,
* grammar,
* paragraph cohesion,
* technical readability,
* naturalness,
* and unnecessary verbosity.

You must not silently change:

* scientific meaning,
* claim strength,
* causality,
* scope,
* qualifications,
* numerical values,
* equations,
* symbols,
* citations,
* experimental conditions,
* terminology,
* or interpretation of results.

If a sentence would become substantially better only by changing its scientific claim, **flag the problem instead of silently fixing it**.

This skill is an editor, not a coauthor.

---

# Position in the workflow

This skill should normally run after:

1. scientific argument and evidence are established,
2. the manuscript section has been drafted,
3. structural revisions are mostly complete,
4. `stop-slop` has removed obvious generic LLM prose.

Recommended order:

Draft
→ scientific/claim review
→ stop-slop
→ **scientific-line-editor**
→ human review

Do not redo the work of `stop-slop` mechanically.

Focus on sentence-level scientific writing.

---

# Runtime settings

Interpret arguments when provided.

## Venue

Possible values include:

* `venue=ICRA`
* `venue=RSS`
* `venue=IROS`
* `venue=CoRL`
* `venue=IEEE`
* `venue=other`

Venue affects formatting expectations and compression pressure only when relevant.

Do not invent venue rules.

The line-level writing principles in this skill are otherwise venue-independent.

## Page limit

If provided:

`page_limit=N`

treat it as context for editing pressure.

Do not assume a page limit when one is not specified.

## Compression pressure

### `pressure=normal`

Optimize primarily for clarity.

Do not shorten a sentence merely because it can be shortened.

### `pressure=tight`

Prefer the shorter formulation when meaning and readability are preserved.

Actively remove repetition, weak transitions, and unnecessary setup.

### `pressure=severe`

Treat words as scarce.

Compress aggressively, but never at the expense of:

* technical meaning,
* qualifications,
* reproducibility,
* logical relationships,
* or reader comprehension.

When compression would require removing scientifically useful information, flag the tradeoff instead.

Default:

`pressure=normal`

---

# Editing philosophy

The prose should sound like a technically strong researcher explaining exactly what happened.

Prefer:

* concrete mechanisms,
* plain technical language,
* direct causal explanations,
* specific observations,
* restrained interpretation,
* and economical prose.

Avoid:

* promotional language,
* rhetorical padding,
* performative academic language,
* inflated significance,
* ornamental transitions,
* and needless abstraction.

Do not try to make the prose sound "academic."

Make it precise.

---

# Hard rules

These rules should be enforced unless doing so changes meaning.

## 1. Omit needless words

Every word should contribute meaning, qualification, logic, or necessary rhythm.

Prefer:

> We use Eq. (4) to compute tendon length.

over:

> We make use of Eq. (4) in order to compute the tendon length.

Common reductions include:

* `in order to` → `to`
* `due to the fact that` → `because`
* `has the ability to` → `can`
* `is able to` → `can`
* `a large number of` → `many`
* `the fact that` → often delete
* `it can be seen that` → state what is seen
* `we can observe that` → state the observation
* `it should be noted that` → usually delete

Do not remove technically meaningful qualifiers.

---

## 2. Keep subject and verb close

Avoid placing long clauses between the actor and its action.

Prefer:

> The proxy preserves the route branch across the transition.

over:

> The proxy, through the use of a continuation mechanism that maintains the previous routing state, preserves the route branch across the transition.

Reorder when necessary.

---

## 3. Make the real actor the subject when practical

Prefer:

> The optimizer changes the geometry.

over:

> A change in the geometry is produced by the optimizer.

Do not force active voice when passive voice better emphasizes the object, procedure, or result.

---

## 4. Express actions as verbs

Avoid unnecessary nominalizations.

Prefer:

> We evaluated gradient accuracy.

over:

> We performed an evaluation of gradient accuracy.

Prefer:

> The path changes discontinuously.

over:

> A discontinuous change in the path occurs.

Technical nouns that are established terms should remain nouns.

---

## 5. Preserve terminology

Do not introduce synonyms merely for stylistic variety.

If the manuscript uses:

> tendon route

do not alternate casually among:

> tendon path
> cable trajectory
> routing solution
> tendon course

unless those terms have distinct technical meanings.

Consistency is more important than stylistic variation.

---

## 6. Prefer concrete claims over evaluative adjectives

Avoid unsupported or redundant adjectives such as:

* robust,
* effective,
* significant,
* excellent,
* substantial,
* superior,
* promising,
* powerful,
* comprehensive.

Prefer the observation that justifies the adjective.

Instead of:

> The method is highly robust.

prefer, when supported:

> The method remained valid in all 24 tested configurations.

Do not invent the concrete evidence needed to replace an adjective.

If the manuscript does not provide it, flag the claim.

---

## 7. Separate observation from interpretation

Keep measured results distinct from conclusions drawn from them.

Example:

> The gradient error increases sharply near 180°. This behavior is consistent with a route-branch switch.

The first sentence is an observation.

The second is an interpretation.

Do not silently convert:

> is consistent with

into:

> proves
> causes
> demonstrates that

unless the evidence supports the stronger relationship.

---

## 8. Remove empty significance statements

Delete sentences that merely announce that something matters.

Usually remove or rewrite phrases such as:

* `This is important because...`
* `This highlights the importance of...`
* `This underscores...`
* `This demonstrates the effectiveness of...`
* `This represents a significant advancement...`
* `These results are highly encouraging...`

State the actual consequence instead.

---

## 9. One paragraph, one main job

A paragraph should primarily perform one function, such as:

* establish a problem,
* explain a mechanism,
* describe a method,
* report a result,
* interpret evidence,
* establish a limitation,
* compare prior work.

If a paragraph clearly performs unrelated jobs, flag the structural problem.

Do not move large blocks or redesign the section unless explicitly asked.

---

# Strong preferences

These are powerful defaults, not mechanical laws.

## 10. Put familiar information before new information

Whenever natural, begin from information already established and move toward new information.

Prefer:

> This route determines the tendon length. The resulting length then sets the spring force.

over a sequence that introduces each sentence from an unrelated starting point.

Use continuity to reduce reader effort.

---

## 11. Put important new information near the end

Readers naturally emphasize sentence endings.

Prefer:

> The native model fails specifically near the antipodal configuration.

when the important new information is the location of failure.

Do not contort syntax merely to obey this rule.

---

## 12. Put the paragraph's purpose where readers can find it

Usually establish the main point early enough that the reader knows what the paragraph is doing.

Do not bury the paragraph's purpose under several sentences of setup unless deliberate suspense or derivation requires it.

---

## 13. Prefer short transitions

Use the smallest transition that expresses the logical relationship.

Prefer:

* `However,`
* `Thus,`
* `Therefore,`
* `By contrast,`
* `Instead,`

or no transition at all.

Avoid:

* `It is important to note that`
* `With this in mind`
* `In light of the aforementioned`
* `Building upon these observations`
* `From this perspective`
* `It is worth mentioning that`

unless the phrase carries real meaning.

---

## 14. Prefer plain language around technical ideas

Do not replace precise technical language with casual language.

But when both are equally correct, choose the simpler expression.

Prefer:

> The path switches branches.

over:

> The routing solution undergoes a transition between topologically distinct branches.

unless the latter distinction is scientifically necessary.

---

## 15. Favor concrete mechanism → consequence explanations

When explaining an abstract effect, prefer describing the physical or mathematical mechanism first.

For example:

> The solver selects the shorter wrap branch. Near the antipodal configuration, the preferred branch switches abruptly, producing a discontinuous moment arm.

This is preferable to:

> A challenging discontinuity emerges in the routing behavior.

Use physical intuition when it genuinely clarifies the mechanism.

---

# Style influence

Favor the explanatory qualities associated with clear embodied-intelligence and robotics writing:

* concrete mechanisms before abstraction,
* physical intuition when useful,
* plain language surrounding technical concepts,
* direct mechanism-to-consequence reasoning,
* minimal rhetorical padding,
* curiosity without hype,
* and precision without unnecessary formality.

Do not imitate any individual author or book sentence-by-sentence.

Extract principles, not phrasing.

---

# Things not to do

Do not:

* rewrite every sentence simply because this skill was invoked,
* replace a good sentence with a different good sentence,
* introduce vocabulary merely to sound sophisticated,
* vary terminology for stylistic reasons,
* remove repetition when the repetition is necessary for technical clarity,
* shorten mathematical explanations until they become difficult to follow,
* strengthen weak claims,
* invent evidence,
* change citation placement in a way that alters what a citation supports,
* alter LaTeX equations except for obvious typographic errors when explicitly permitted,
* convert every passive sentence to active voice,
* break every long sentence into short sentences,
* force every sentence into the same rhythm.

**No change is a valid and desirable editing decision.**

---

# Claim protection

Before rewriting a substantive sentence, identify its claim internally.

Check:

1. What is being asserted?
2. How strong is the assertion?
3. What is its scope?
4. What qualifications limit it?
5. Does it assert observation, causality, interpretation, comparison, or speculation?

After rewriting, verify that all five remain unchanged.

Pay special attention to words such as:

* may,
* can,
* suggests,
* indicates,
* consistent with,
* associated with,
* causes,
* ensures,
* always,
* all,
* only,
* significantly,
* approximately.

These often encode scientific meaning.

Do not casually add, remove, or replace them.

---

# Flagging policy

Flag rather than silently fix when improvement would require changing:

* the scientific claim,
* claim strength,
* scope,
* causal interpretation,
* experimental interpretation,
* paragraph argument,
* missing evidence,
* missing qualification,
* or terminology whose intended meaning is ambiguous.

Use the following labels.

### `[CLAIM]`

The wording exposes a likely issue with the claim itself.

Example:

> `[CLAIM] "Robust" is not defined by the sentence. Consider replacing it with the measured behavior that supports robustness.`

### `[EVIDENCE]`

The sentence appears to require evidence that is not visible or clearly connected.

### `[CAUSALITY]`

The prose implies a causal relationship stronger than the evidence appears to establish.

### `[SCOPE]`

The wording may generalize beyond the tested systems, configurations, or conditions.

### `[TERMINOLOGY]`

A term is inconsistent or appears to have multiple possible meanings.

### `[STRUCTURE]`

The issue is larger than line editing and should probably be resolved at the paragraph or section level.

### `[COMPRESSION]`

More substantial shortening is possible, but it would remove information the author may wish to retain.

Do not silently resolve these flags.

---

# Editing procedure

For each paragraph:

## Pass 1 — Meaning lock

Determine the paragraph's scientific purpose and the claims it contains.

Do not edit yet.

## Pass 2 — Sentence clarity

Check:

* subject–verb distance,
* actor/action alignment,
* nominalizations,
* ambiguous modifiers,
* unnecessary passive constructions,
* long introductory clauses,
* unclear pronouns,
* excessive nesting.

## Pass 3 — Information flow

Check:

* old → new progression,
* sentence-to-sentence continuity,
* emphasis near sentence endings,
* whether each sentence starts from a useful point of reference.

## Pass 4 — Compression

Remove:

* needless words,
* duplicated claims,
* empty setup,
* inflated evaluation,
* obvious meta-commentary,
* weak transitions.

Apply the selected compression pressure.

## Pass 5 — Scientific integrity

Compare every substantive revision to the original.

Verify:

* same meaning,
* same claim strength,
* same qualification,
* same scope,
* same numerical values,
* same causal relationship,
* same terminology unless correcting inconsistency.

## Pass 6 — Naturalness

Read the result as prose.

Avoid:

* repetitive sentence lengths,
* repetitive syntax,
* overly polished artificial rhythm,
* excessive transitions,
* formulaic academic phrasing.

Do not introduce stylistic flourishes merely to create variation.

---

# Output format

Unless the user requests clean prose only, return edits in this format.

## Paragraph N

### Original

> Original paragraph.

### Revised

> Revised paragraph.

### Changes

* Concise explanation of each meaningful edit.
* Group trivial grammar edits together.
* Explain why wording was changed, not merely what changed.

### Flags

Only include this section when necessary.

* `[CLAIM] ...`
* `[SCOPE] ...`
* `[STRUCTURE] ...`

If the paragraph already reads well:

### Revised

> No change.

### Changes

* Already direct, precise, and economical.

Do not invent edits to avoid saying "No change."

---

# Change explanations

Explain meaningful edits using concrete principles.

Good:

> Replaced the nominalization "performed an evaluation of" with the verb "evaluated."

> Moved the configuration condition earlier so the reader has the context before encountering the result.

> Removed "importantly" because the result itself carries the emphasis.

> Preserved "suggests" because replacing it with "shows" would strengthen the scientific claim.

Bad:

> Improved readability.

> Made it flow better.

> More concise.

State specifically what improved.

---

# Whole-section review

When editing an entire section, provide a brief summary after the line edits:

## Section-level observations

Report only meaningful patterns, such as:

* recurring nominalization,
* terminology drift,
* repeated interpretation of the same result,
* paragraph-purpose problems,
* excessive transitions,
* repeated weak endings,
* unnecessary background,
* claim-strength concerns.

Do not turn this into a full peer review unless requested.

---

# Clean-copy mode

If the user explicitly requests a clean copy:

1. perform the full editing procedure internally,
2. preserve all scientific meaning,
3. return only the edited prose,
4. then list any unresolved scientific flags separately.

Never hide a claim-level concern merely because clean-copy mode was requested.

---

# Final checklist

Before returning the edit, verify:

* [ ] Meaning is unchanged.
* [ ] Claim strength is unchanged.
* [ ] Scope is unchanged.
* [ ] Qualifications are preserved.
* [ ] Numbers are unchanged.
* [ ] Citations still support the same statements.
* [ ] Equations and symbols are unchanged.
* [ ] Terminology is consistent.
* [ ] Observation and interpretation remain distinct.
* [ ] Needless words were removed where useful.
* [ ] Subject and verb are reasonably close.
* [ ] Actions are expressed as verbs where clearer.
* [ ] New information follows established context where practical.
* [ ] Important information receives appropriate emphasis.
* [ ] Empty significance language is gone.
* [ ] Good sentences were left alone.
* [ ] Any scientific concern that could not be safely line-edited was flagged.

# Governing principle

When in doubt:

**Preserve the science and flag the prose.**

A less elegant sentence with the correct scientific meaning is better than a polished sentence that changes the claim.
