# Paper Writing Guide

> **Purpose:** Canonical writing principles for turning approved science into a clear technical paper.
>
> **Scope:** This guide governs prose and narrative presentation. It does not determine scientific truth, novelty, experiment validity, or claim approval.
>
> **Core rule:** **Write the paper the evidence supports. Do not improve the science with prose.**

---

# 1. The paper needs one clear scientific spine

Before drafting, reduce the paper to:

> existing knowledge  
> → unresolved problem  
> → what we did  
> → what we found  
> → what we derived or explained  
> → why it matters

Every major section should advance this chain.

If a paragraph does not support the paper's approved story, ask whether it belongs.

The approved version of this chain lives in:

`research/10_PAPER_STORY.md`

Do not invent a different story while drafting.

---

# 2. Separate literature, gap, action, result, and implication

A reader should be able to tell whether a sentence describes:

- prior literature,
- an unresolved question,
- our action,
- our result,
- our interpretation,
- or a limitation.

Preferred pattern:

> Prior work establishes X. However, it does not establish Y. We therefore perform A. We show B. We derive C. This enables D.

Avoid compressing all of these roles into one ambiguous sentence.

---

# 3. Use explicit contribution ownership

When describing this project's work, prefer direct `We + action verb` constructions when ownership would otherwise be unclear.

Useful verbs:

- We develop
- We implement
- We formulate
- We validate
- We verify
- We compare
- We isolate
- We test
- We measure
- We show
- We find
- We derive
- We characterize
- We quantify
- We confirm
- We evaluate

Do not mechanically begin every sentence with `We`.

Use active contribution language where it makes clear **what this paper did**.

Weak:

> Forward agreement does not guarantee gradient fidelity.

Stronger:

> We compare alternative route models under matched forward geometry and show that forward agreement does not guarantee gradient fidelity.

---

# 4. Do not make new results sound like established background

Keep literature and contribution boundaries explicit.

Preferred:

> Prior work shows that differentiable simulators can produce misleading gradients. However, it remains unclear whether alternative route-resolution choices can disagree in gradient direction or scale despite matched forward geometry. We test this directly.

Avoid writing the project's own result as though the literature already established it.

---

# 5. Describe prior work neutrally

Prefer:

> Prior work establishes X but does not establish Y.

over:

> Prior work fails at Y.

Useful gap language:

- does not establish
- has not isolated
- leaves unresolved
- does not characterize
- has not evaluated
- does not distinguish
- has not tested under matched conditions

Do not write absolute novelty claims unless the literature evidence supports them.

Prefer:

> To our knowledge, prior work has not...

over:

> No prior work has...

when appropriate.

---

# 6. Name the model, method, or comparison

Avoid vague subjects such as:

- the distortion
- this disagreement
- the failure
- the reference
- these models

when the reader could reasonably ask "which one?"

Weak:

> The same distortion affects design gradients.

Stronger:

> We find that Hard Shortest and Smooth Relaxation also produce scale-distorted design gradients relative to the Continuation reference.

Specific writing is usually stronger than dramatic writing.

---

# 7. Distinguish the framework from the baselines it evaluates

When comparing methods, explicitly distinguish:

- our framework or implementation,
- the reference,
- baselines,
- learned approximations,
- native evaluators,
- which representation produces which behavior.

Do not write generic statements that make a baseline limitation sound like a failure of the whole framework.

---

# 8. Be precise about what is actually ours

Do not claim prior concepts as novel merely because this project implements them.

Prefer:

> our validated Continuation implementation

over:

> our Continuation method

when the underlying principle comes from prior work.

Ownership may apply to:

- implementation,
- validation,
- controlled comparison,
- derivation,
- experiment,
- analysis,
- use as a reference.

---

# 9. Contribution bullets must state actions

A contribution bullet should answer:

> What did this paper do that matters?

Preferred structure:

> **Contribution name.** We do X. We show Y. We derive or validate Z.

Avoid beginning a contribution bullet with a generic observation that hides the paper's action.

---

# 10. Do not headline the paper with its own failure cases

When a result identifies a problematic condition, name the scientific object rather than framing it as a failure of the project.

Prefer:

- geometric feasibility
- route-switch geometry
- route regularity
- design admissibility
- direction error
- scale error
- transfer discrepancy

over generic labels such as:

- Failure A
- Failure B
- method failure
- catastrophic failure

unless "failure" is scientifically precise and intentionally used.

---

# 11. Separate observation from explanation

Use this structure:

> Observation → test of explanation → evidence → consequence

Example:

> We observe gradient disagreement. We derive a route-switch manifold and evaluate conditions ranked by distance to it. The disagreement increases near the manifold. This produces measurable changes in local updates.

Do not convert correlation into causation without evidence.

Prefer:

- associated with
- increases near
- organizes
- is consistent with

unless the evidence justifies stronger causal language.

---

# 12. Preserve limitations explicitly

A limitation is not a weakness in writing if it is scoped correctly.

Good:

> Native finite differences confirm the gradient direction predicted by the reference, though not its magnitude.

Distinguish:

- contribution,
- evidence,
- scope,
- limitation.

Do not erase a limitation simply because it makes the sentence less assertive.

---

# 13. Scope language is good scientific writing

Use explicit scope qualifiers:

- in the tested mechanism
- over the tested operating range
- across the evaluated builders
- for the five predeclared rollouts
- in the monoarticular case
- for the evaluated seeds

Do not generalize beyond the evidence.

---

# 14. Abstract structure

The abstract should usually follow:

1. broad field context,
2. more specific background,
3. unresolved problem,
4. what we do,
5. main result,
6. extension or mechanism,
7. broader significance.

A useful pattern is:

> Here, we present...  
> We show...  
> We further show...  
> We derive...  
> We use...  
> These results indicate...

Avoid dense sentences that contain multiple contributions joined by semicolons.

---

# 15. Introduction structure

A strong Introduction usually has four jobs:

### Paragraph 1 — field and modeling landscape
What approaches exist and why the problem matters.

### Paragraph 2 — unresolved issue
What prior work establishes and what remains unclear.

### Paragraph 3 — study question / thesis
What this paper asks.

### Paragraph 4 — contributions
What we did and established.

Do not write a miniature Related Work section in the Introduction.

Do not hide the actual gap until late in the section.

---

# 16. Related Work structure

Organize by:

- question,
- modeling family,
- methodological family,
- or conceptual distinction.

Avoid chronological paper-by-paper narration.

Use literature to establish:

> what is already known → where this work differs

Do not repeatedly announce novelty.

---

# 17. Methods structure

Methods should expose the conceptual structure needed to understand the results.

- Define symbols promptly.
- Make analytically distinct families visibly distinct.
- Explain the minimum machinery necessary.
- Do not narrate implementation chronology.
- Do not present code organization as a scientific contribution.

---

# 18. Results structure

Every experiment should answer a question.

Preferred sequence:

> question  
> → experimental design  
> → result  
> → interpretation  
> → boundary

State what the metric means.

Weak:

> Cosine similarity was -0.258.

Stronger:

> Hard Shortest produces a gradient in the opposite direction from the reference (cosine similarity -0.258).

---

# 19. Discussion structure

Discussion should:

- connect results,
- interpret mechanisms,
- state where the method matters,
- explain scope,
- preserve limitations.

Do not simply repeat numerical results.

A good pattern is:

> We show X. This means Y. The likely mechanism is Z. The scope is A. The limitation is B.

---

# 20. Conclusion structure

The Conclusion should state what was established.

A strong conclusion often follows:

> We developed A.  
> We used A to show B.  
> We derived C.  
> We used C to quantify D.

Do not introduce new claims.

Do not turn the conclusion into marketing copy.

---

# 21. Prefer quantitative evidence over dramatic adjectives

Prefer:

> transfer error is 80× the threshold

over:

> transfer fails catastrophically.

Prefer:

> gradient magnitude differs by 1229×

over:

> the gradient is extremely inaccurate.

Let measurements carry emphasis.

---

# 22. Separate gradient direction and magnitude

When gradients matter, distinguish:

- direction,
- component sign,
- magnitude,
- update consequence.

A single aggregate gradient-error number can hide important behavior.

---

# 23. Avoid research-history narration

Do not tell the reader how the project happened chronologically unless the chronology matters scientifically.

Avoid:

- first we tried...
- at one arbitrary point...
- this left open whether...
- we then happened to test...

Prefer the final logical experiment:

> To test whether gradient fidelity depends on route-switch proximity, we evaluate five predeclared operating points ranked by exact geometric distance.

---

# 24. One paragraph, one primary job

A paragraph should mainly:

- establish background,
- state a gap,
- explain a method,
- report a result,
- interpret evidence,
- establish a limitation,
- or compare prior work.

If a paragraph performs several unrelated functions, restructure it.

---

# 25. Prefer plain technical language

Prefer:

> The path switches branches.

over:

> The routing solution undergoes a transition between topologically distinct branches.

unless the stronger terminology is scientifically necessary.

Do not try to make prose sound "academic."

Make it precise.

---

# 26. Preserve terminology

Do not vary technical terminology merely for style.

If `route semantics` is the chosen term, do not alternate casually among:

- route logic,
- path policy,
- route selection rule,
- routing behavior

unless those terms mean different things.

Consistency is more important than stylistic variety.

---

# 27. Remove empty significance language

Avoid:

- importantly
- notably
- this highlights the importance
- this underscores
- this represents a significant advancement
- promising
- powerful
- comprehensive

State the actual consequence instead.

---

# 28. Figures should have one sentence of purpose

Before creating or revising a figure, define:

> The reader should learn that ______ from this figure.

Every panel should support that statement.

Do not let figure design drive the science.

After figure freeze, do not regenerate figures during prose revision unless explicitly authorized.

---

# 29. Distinguish audit from edit

Use separate tasks.

### Audit
Identify problems. Do not edit.

### Revision
Apply only approved changes.

Do not ask an agent to "improve the paper" late in the process.

---

# 30. Late-stage revision must be surgical

For final revisions:

- define allowed files,
- define forbidden files,
- preserve science,
- preserve figures unless explicitly changing one,
- compile,
- inspect diff,
- verify page count,
- do not commit unless asked.

Use `strict-paper-revision`.

---

# 31. Final consistency test

Before submission, independently compare:

1. Abstract
2. Introduction gap
3. Contribution bullets
4. Results conclusions
5. Discussion opening
6. Conclusion

They should agree on:

- prior knowledge,
- gap,
- action,
- result,
- responsible method/model,
- scope,
- limitation.

If they tell different versions of the paper, fix the inconsistency.

---

# 32. Governing principle

When writing or editing:

> **Preserve the science and improve the communication.**

When a cleaner sentence requires changing the claim:

> **flag the claim instead of silently rewriting it.**
