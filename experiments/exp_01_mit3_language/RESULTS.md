# EXP-01 Results - MIT experiment 3 replication (reading a language model's middle blocks)

## Experiment
Replication of Experiment 3 of "Sparks of Weight-Space Programming" (Takehi, Teoh, Zhang, Khattab, MIT CSAIL,
2 Nov 2026). Protocol source: `external/lm_understand-logs/exp3_language/README.md`. No experiment contract
was written beforehand; the researcher asked for the replication directly (2026-11-21).

## Run status
COMPLETE (6 of 6 conversations finished with a final answer).

## What actually ran
- Reader: `gpt-6-astra` through `codex exec` (codex-cli 0.169.0-alpha.11.1 bundled in ChatGPT.app), reasoning
  effort `xhigh`, one `codex exec` call per turn, `codex exec resume` for later turns.
- Networks: MIT's released middle blocks (`networks/<language>/blocks.npz`, 7,084,032 numbers each) of three
  4-layer x 6-head, width-384 character-level transformers trained on Python code, romaji Japanese or English
  Wikipedia. Embeddings and output layer are not released and were never available.
- First message: MIT's `message.txt`, verbatim. Reply format returned to Astra: `OUTPUT of run k of at most 100:`
  followed by the printed output in a fenced block, as in MIT's logs.
- Code execution: Astra replied with one Python block per turn; we ran it in a macOS `sandbox-exec` sandbox
  (no network, no reads in the home directory outside the sandbox environment and the work folder, fresh process
  and emptied folder per run, 600 s limit, output cut at 20,000 characters). Up to 100 runs allowed.
- Two independent conversations per network (`_r0`, `_r1`), three conversations in parallel.
- Date: 2026-11-21. Total Codex wall time summed over turns: 3,499 s.

## Deviations from the source protocol
- Sandbox is macOS `sandbox-exec`, not bubblewrap; packages available: numpy, scipy, scikit-learn, PyTorch
  (CPU), pandas, sympy, networkx (same list as MIT).
- Codex tools: MIT switched every Codex tool off. With this Codex build some tool definitions cannot be
  switched off from the command line (`functions.exec`, `functions.wait`, `request_user_input`, `collaboration.*`
  remain listed to the model). Every turn's event stream was audited instead: all 54 accepted turns contain
  only model messages (`tool_free: true`), so no Codex tool was used.
- The user's global Codex `AGENTS.md` (style preferences, about 1 KB) is still appended to the instructions;
  it could not be disabled without changing the user's Codex home. MIT used the single line
  "Answer the user's request." (we pass the same line as the model instructions file).
- Ground truth was not re-measured: it is MIT's `eval/heads.json` (measured by MIT by running the full
  networks, which are not released).

## Artifacts
- `raw/runs/<run>/` - per conversation: `message.txt`, `turns/tNNN/{prompt.txt,answer.md,events.jsonl,run.json}`,
  `tool_calls.jsonl` (every code run with its output), `final_answer.md`, `state.json`, `loop.log`
- `processed/scores.json` - parsed answers and scoring (written by `score.py`)
- `run.py`, `score.py`

## Numerical results
Copying head = induction score > 0.2 in MIT's `eval/heads.json` (the cutoff the post uses).

| Run | Truth | Our Astra: corpus (P) | Copy heads named | True copy heads | False positives | Missed | Code runs | MIT's Astra: corpus (P) | MIT copy heads | MIT code runs |
|---|---|---|---|---|---|---|---|---|---|---|
| python_r0 | python | python (0.65) | 2.1, 3.0, 3.1, 3.5 | 2.1, 3.1, 3.5 | 3.0 | - | 10 | python (0.56) | 2.1, 3.1, 3.5 | 11 |
| python_r1 | python | python (0.70) | 2.1, 3.0, 3.1, 3.5 | 2.1, 3.1, 3.5 | 3.0 | - | 9 | python (0.52) | 2.1, 3.1, 3.5 | 7 |
| romaji_r0 | romaji | romaji (0.58) | 3.3 | 3.3 | - | - | 6 | romaji (0.55) | 3.3 | 8 |
| romaji_r1 | romaji | romaji (0.60) | 3.3 | 3.3 | - | - | 9 | romaji (0.62) | 3.3 | 9 |
| english_r0 | english | english (0.55) | 3.1 | 3.1 | - | - | 6 | english (0.58) | 3.1 | 7 |
| english_r1 | english | english (0.55) | 3.1 | 3.1 | - | - | 7 | english (0.48) | 3.1 | 6 |

- Corpus identified correctly: 6 of 6 (MIT: 6 of 6).
- True copying heads recovered: 10 of 10 across the six conversations (MIT: 10 of 10).
- Conversations with exactly the true set of copying heads: 4 of 6 (MIT: 6 of 6). Both Python conversations
  additionally label head 3.0 as `copy`; its measured induction score is 0.018 (the largest score of any
  non-copying head in that network is 0.019). In `python_r0` the label is marked "tentative".
- All 24 heads received a label in 6 of 6 conversations.

## Observations
- `[OBSERVATION]` Corpus identification and the set of true copying heads reproduce with our own Astra access.
- `[OBSERVATION]` Our Python conversations over-call one head (3.0) that MIT's two Python conversations did not.

## Negative / null results
- None beyond the two false-positive head labels above.

## Failed / excluded runs
- `english_r1`, turn 1, attempt 1: Codex returned no final message (rc 0). The harness waited 60 s and asked
  again in a new conversation; the retry is the conversation reported. The failed attempt is kept in
  `raw/runs/english_r1/turns/t001_fail1/`.

## Anomalies
- A keyword scan of all 47 code runs for forward-pass operations found `np.exp` only inside an entropy-based
  effective-rank formula, and "ReLU" only in printed disclaimers. No run executes the network or a layer on an
  input. This was a keyword scan plus reading the flagged snippets, not a line-by-line review of every run.

## Scope
Three networks released by MIT, two conversations each, one model (`gpt-6-astra`, `xhigh`), one prompt.

## What this experiment does NOT establish
- It does not independently verify MIT's ground truth (the full networks are not released).
- Two conversations per network do not estimate how often the corpus or head labels would be right.
- It says nothing about larger models or about networks trained by us.

## Reproducibility
- command: `python experiments/exp_01_mit3_language/run.py --parallel 3` then `python experiments/exp_01_mit3_language/score.py`
- inputs: `external/lm_understand-logs` (clone of github.com/RikiyaT/lm_understand-logs, not in git)
- environment: macOS, Apple M4 Max; sandbox environment `.venv-sandbox` (Python 3.12, numpy 2.5.5, torch 2.15.0)
- seeds: none (Astra's sampling is not controllable through Codex)

## Status
Awaiting researcher review.
