# Research Paper Template

A reusable research-paper workspace for **researcher-directed science with AI-assisted memory, implementation, figure production, ghost writing, revision, and submission checks**.

Claude helps execute and communicate the research. The researcher decides:

- what questions matter;
- which experiments are scientifically valid;
- what results mean;
- which claims are approved;
- what the paper's story is.

---

## Workflow

```text
Problem / phenomenon
→ field map
→ candidate questions / gaps
→ approved experiment contracts
→ experiment results
→ approved ground truth
→ claim–evidence map
→ approved paper story
→ drafting
→ review / revision
→ submission audit
```

For the full process, see:

- `docs/guidelines/WORKFLOW.md`
- `docs/guidelines/PAPER_WRITING_GUIDE.md`

---

## Repository map

| Path | Purpose |
|---|---|
| `CLAUDE.md` | Claude orchestration, authority, and routing rules |
| `research/` | Durable scientific state and paper reasoning |
| `docs/intakes/` | One historical intake per research session |
| `docs/CURRENT_STATE.md` | Temporary active-task checkpoint |
| `docs/guidelines/` | Human-readable workflow and writing guidance |
| `.claude/agents/` | Bounded delegated roles |
| `.claude/skills/` | Repeatable procedures |
| `experiments/` | Experiment code, configs, outputs, and `RESULTS.md` |
| `paper/` | Manuscript, venue configuration, figures, bibliography |
| `paper/vendor/` | Immutable external venue/template references |

---

## Scientific source of truth

When project files disagree, check in this order:

1. Current explicit researcher instruction
2. `research/07_RESULTS_GROUND_TRUTH.md`
3. `research/08_CLAIM_EVIDENCE_MAP.md`
4. `research/09_DECISION_LOG.md`
5. `research/10_PAPER_STORY.md`
6. Approved experiment contracts / experiment `RESULTS.md`
7. `research/00_STATE_OF_PROJECT.md`
8. Literature, gap, and hypothesis files
9. Manuscript prose
10. Session intakes / exploratory notes

**The manuscript is not the scientific source of truth.**

---

## Claude components

### Agents

- `scout` — locate code, files, dependencies, tests
- `implementer` — implement approved coding tasks
- `source-reader` — extract evidence from supplied sources
- `evidence-auditor` — check whether claims are supported
- `writer` — ghost-write approved science
- `verifier` — independently verify outputs

### Skills

- `checkpoint`
- `project-state-update`
- `experiment-contract`
- `result-handoff`
- `figure-production`
- `paper-drafting`
- `embodied-ai-paper-writer`
- `stop-slop`
- `scientific-line-editor`
- `strict-paper-revision`
- `submission-audit`

Each skill and agent documents its own behavior. `CLAUDE.md` explains when to use them.

---

## External components

Third-party writing skills are included as git submodules:

- `.claude/skills/stop-slop`
- `.claude/skills/embodied-ai-paper-writer`

After cloning:

```bash
git submodule update --init --recursive
```

Or clone with:

```bash
git clone --recurse-submodules <repo-url>
```

Venue/template reference material lives under:

```text
paper/vendor/
```

Do not edit vendor files.

---

## If something breaks

| Problem | Check first |
|---|---|
| Claude cannot find a skill | `.claude/skills/<skill>/SKILL.md` |
| External skill directory is empty | `.gitmodules`, then initialize submodules |
| Claude uses the wrong research fact | `07_RESULTS_GROUND_TRUTH.md` → `08_CLAIM_EVIDENCE_MAP.md` |
| Project-level research state is stale | `project-state-update` → `research/00_STATE_OF_PROJECT.md` |
| A task cannot be resumed | `docs/CURRENT_STATE.md` → `checkpoint` |
| Experiment intent is unclear | `research/06_EXPERIMENT_CONTRACTS.md` |
| Experiment result provenance is unclear | experiment `RESULTS.md` → `07_RESULTS_GROUND_TRUTH.md` |
| Figure changes unexpectedly when regenerated | `figure-production` generator/asset synchronization rules |
| Figure style is inconsistent | `.claude/skills/figure-production/ZLabPlot_STYLE_GUIDE.md` |
| Writing changes scientific meaning | `08_CLAIM_EVIDENCE_MAP.md`, `paper-drafting`, `scientific-line-editor` |
| Late revision changes unrelated content | `strict-paper-revision` |
| Paper story becomes inconsistent | `research/10_PAPER_STORY.md` |
| Venue formatting is unclear | `paper/VENUE.md` + `paper/vendor/` |
| Submission fails compliance | `submission-audit` |
| Claude behavior is globally wrong | `CLAUDE.md` |
| You do not remember why a decision was made | `research/09_DECISION_LOG.md` |

---

## Updating the template

Change the most local source of behavior:

| Change needed | Update |
|---|---|
| Scientific-state schema | `research/*.md` |
| Research/paper process | `docs/guidelines/WORKFLOW.md` |
| Writing principles | `docs/guidelines/PAPER_WRITING_GUIDE.md` |
| Repeatable task procedure | `.claude/skills/<skill>/SKILL.md` |
| Delegated role | `.claude/agents/<agent>.md` |
| Figure style/tooling | `.claude/skills/figure-production/` |
| Venue requirements | `paper/VENUE.md` |
| External template reference | `paper/vendor/` |
| Global Claude orchestration | `CLAUDE.md` |
| Quick navigation / troubleshooting | `README.md` |

After changing a component, check that `CLAUDE.md` and this README still reference the correct paths and names.

---

## Maintenance rule

Keep this README short.

It should answer:

> **What is this system, where does each responsibility live, and where do I look when something breaks?**

Detailed behavior belongs in the referenced files, not here.

**Last template audit:** `YYYY-MM-DD`
