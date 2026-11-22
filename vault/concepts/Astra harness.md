---
type: concept
tags: [concept, harness, astra]
---

# Astra harness - how we run GPT-6 Astra through Codex

Code: `wsp/harness/` (`codex_client.py`, `sandbox.py`, `loop.py`). It mirrors the harness MIT released with
[[Takehi et al. 2026 - Weight-Space Programming]] (`external/lm_understand-logs/harness/`).

## One turn = one `codex exec` call
- Binary: `/Applications/ChatGPT.app/Contents/Resources/codex-cli/bin/codex` (0.169). The standalone CLI in
  `~/.local/bin` (0.144.1) is too old: the API answers "gpt-6-astra requires a newer version of Codex".
- Flags: `--ignore-user-config --ignore-rules --skip-git-repo-check --json -m gpt-6-astra -s read-only`, effort
  set per call (`xhigh` for reading experiments, `max` for MIT exp 8), instructions file = the single line
  "Answer the user's request.", every tool-giving feature set to false.
- The prompt goes in on stdin. Later turns use `codex exec resume <thread id>`.
- No API key is involved; this uses the ChatGPT login of the Codex app.

## Tool-free by audit
Some tool definitions (`functions.exec`, `collaboration.*`, `request_user_input`) cannot be removed from the
command line in this build, and the user's global `~/.codex/AGENTS.md` is still appended. So each turn's
event stream is audited: a turn is accepted only if it contains nothing but model messages and reasoning. A turn
that used a tool is discarded and re-asked with a reminder.

## How Astra acts
Astra replies with exactly one fenced block. The loop runs it and sends the printed output back as the next
message ("OUTPUT of run k of at most N:" + fenced output). A reply without a block is the final answer.
Block types are per experiment: `python` (sandboxed analysis), and later `forward` / `attention` tools.

## Sandbox for Astra's code
macOS `sandbox-exec`: no network; writes only in the run's work folder; no reads inside the home directory
except the sandbox Python environment (`.venv-sandbox`: numpy, scipy, scikit-learn, torch CPU, pandas, sympy,
networkx) and the work folder; `python -I`; time limit; output cap. Work folders have neutral names so a path
cannot leak which network is being read.

## Logs
Same layout as MIT's: `turns/tNNN/{prompt.txt,answer.md,events.jsonl,run.json}`, `tool_calls.jsonl`,
`final_answer.md`, `state.json`, `loop.log`. Local paths are replaced by `<repo>` and `<home>`.

## Links
[[Weight-space programming]] - [[EXP-01 MIT exp 3 - reading a language model's middle blocks]]
