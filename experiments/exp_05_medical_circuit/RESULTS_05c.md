# EXP-05c Results - Donor session `transfer1`: Astra copies MediPhi-Instruct weights into Phi-3.5 (bare-prompt MedQA)

Part b of EXP-05 (activation patching and per-layer weight swapping) is in `RESULTS.md` in this folder.

## Experiment
Filippos Bellos's main direction (2026-11-21): use GPT-6 Astra to find which components make the medical
fine-tune succeed where the base model fails, and move that capability into the base model by targeted weight
edits, without fine-tuning. No formal experiment contract was written. The working specification is
`vault/experiments/EXP-05 Medical circuit discovery and transfer.md` and the task text sent to Astra
(`raw/transfer1/message.txt`). Decided by the researcher on 2026-11-22: donor = `microsoft/MediPhi-Instruct`,
primary measure = accuracy under the bare lm-evaluation-harness prompt, guardrail = accuracy under the chat prompt
must not fall, and the test of success is the MedQA test split under both prompts. The budget (30 forward
passes), the effort level and the edit restrictions are Claude's provisional choices.

## Run status
COMPLETE. Astra used all 30 forward passes, withdrew its candidate edit and submitted the unedited model. The
withdrawn candidate was scored on the test split afterwards, at the researcher's request (section "The withdrawn
candidate on the test split").

## What actually ran
- Model edited: `microsoft/Phi-3.5-mini-instruct`. Donor: `microsoft/MediPhi-Instruct`. Both float16 on Apple
  MPS. Editor: `gpt-6-astra` through `codex exec`, effort `xhigh`, all Codex tools off, every turn audited as
  tool-free.
- Astra's material: both sets of weights in a sandboxed CPU-only Python kernel; 4,272 practice questions (MedQA
  dev 1,272 + the first 3,000 training questions) with answers and both models' results under both prompts; 30
  forward passes. A pass is one of: scoring up to 1,024 questions under one prompt (with paired statistics when
  the unedited and the edited model are scored together), a teacher-forced read of activations, or activation
  patching from the donor on up to 64 questions.
- Prompts: "bare" = "Question: ...\nA. ...\nB. ...\nC. ...\nD. ...\nAnswer:" as plain text; "chat" = the chat
  prompt of EXP-04 Protocol A. The answer is the letter with the highest next-token probability. Margin =
  log-probability of the correct letter minus the best wrong letter.
- Session: 69 replies from Astra, 36 kernel runs, 32 forward requests of which 30 were counted (22 score passes,
  5 teacher-forced reads, 3 patching passes). One conversation restart with the notebook at reply 41. Started
  2026-11-22 23:31, final answer 2026-11-23 14:09. Astra's own time: 3,060 s. Time inside forward passes:
  13.8 h, most of it waiting (see Deviations).
- Astra's own protocol (its choice; `work/notes.md`, written before any forward pass): it split the practice
  pool with seed 815731 into 1,024 confirmation questions, set aside, and 3,248 development questions. Inside the
  development questions it used 64 "discovery" questions for activations and patching (24 only the donor right,
  16 only the base right, 12 both, 12 neither), 256 "exploration" questions to compare transfers, and 512
  "validation" questions as a second check.
- Astra's retention rule, written in its notes after pass 23 and in `work/candidate_manifest.json` (file time
  13:20) before the confirmation passes (13:40 to 14:07): keep the candidate only if the bare-prompt accuracy gain
  on the 1,024 confirmation questions exceeds 2 paired standard errors and the chat-prompt accuracy difference is
  not negative; otherwise submit no edit, with no fallback chosen from the confirmation results.
- Final measurement by the harness: the submitted weights against the unedited model on the MedQA test split
  (1,273 questions, never shown to Astra) under both prompts, paired. 2026-11-23 14:09 to 14:22.
- Checks by Claude afterwards (`check_session.py`, 14:28 to 14:39): recount of every score pass, comparison of
  the saved candidate with both checkpoints, and a second scoring of the saved candidate on the confirmation
  questions.
- After the session, asked for by the researcher on 2026-11-23: the saved candidate against the unedited model on
  the MedQA test split under both prompts (`check_session.py --test`), same scoring code, batches of 16. What to
  report (paired statistics, Astra's rule, the split by chat result, the donor's own results) was written into
  the script before the run started. 15:05 to 16:58.

## Deviations from contract
- No contract exists.
- The researcher's test of success (test split, both prompts) was applied by the harness to the submitted
  weights only, which are the unedited weights. The withdrawn candidate was scored on the test split afterwards.
  That run was not part of the session: Astra never saw it and its decision did not depend on it.
- The test-split run took 1 h 53 min for about 13 minutes of computing: the Mac was asleep with the lid closed
  for most of the time between 15:07 and 16:35.
- Waiting. The session shared the GPU with the EXP-05b and EXP-07 sweeps and the EXP-03 Qwen run. Pass 6 (a
  patching pass, which needs both GPU locks) waited from 02:33 to 09:07 because the EXP-07 scan held one lock for
  its whole run; passes 3 and 4 (23:36 to 01:11 and 01:13 to 02:25) overlapped the EXP-05b runs, which held GPU
  locks from 23:32 to 00:41 and from 01:11 to 02:20. The Mac was asleep from 03:32 to 09:02 (idle sleep),
  from 10:07 to 11:17 (battery empty, hibernation) and for shorter periods with the lid closed (09:48 to 09:49,
  09:52 to 09:56, 11:40 to 11:59, 12:00 to 12:12, 12:26 to 12:32). The process continued from the same state each
  time.
- Code change during the session: commit `69a1948` (12:14) changed how `wsp/lab.py` takes the GPU locks. The
  session's parent process kept the code it had loaded at 23:31; worker processes started after 12:14 imported the
  new file. The scoring path (`MedQALab._score`, `wsp/medqa.py`, `wsp/patching.py`) did not change.
- Two forward requests were rejected before any model ran ("a score pass takes 1 to 1024 questions"; Astra's
  notes say the request listed question ids twice). They were not counted against the budget.

## Artifacts
- `raw/transfer1/message.txt` - the task text. `raw/transfer1/turns/tNNN/` - every prompt, reply and Codex event
  stream. `raw/transfer1/tool_calls.jsonl`, `raw/transfer1/forward_calls.jsonl` - every kernel run and forward
  request with output. `raw/transfer1/final_answer.md` - Astra's final answer. `raw/transfer1/final_eval.json`,
  `final_eval_per_question.jsonl` - the harness's test-split measurement.
- `check_session.py` - Claude's checks. `processed/check_transfer1.json` - their result (per-pass recount, weight
  comparison per tensor, the second scoring). `processed/check_transfer1_questions.jsonl` - the per-question
  result of all 22 score passes, and the donor's EXP-04 result on the confirmation questions. The `test` block of
  `check_transfer1.json` and `processed/check_transfer1_test_questions.jsonl` hold the test-split run.
- Viewer dataset `exp05c_transfer_transfer1` (`python -m wsp.viz_export exp05c`; data under `viz/public/data/`,
  not in git); screenshots `vault/attachments/viz_exp05c_transfers.jpg`, `viz_exp05c_candidate_weights.jpg`,
  `viz_exp05c_per_question.jpg`.
- Not in git (the `work` folder is ignored; it stays on this machine at `raw/transfer1/work/`): `notes.md`
  (Astra's notebook), `final_report.json`, `candidate_manifest.json`, the split id files, `score_<k>.jsonl` and
  `patch_<k>.jsonl` (the lab's per-question output of each pass), `fwd_<k>/` (activation arrays), and
  `candidate_sparse_full/` (the candidate's 46 tensors, 3.5 GB).

## Numerical results

### Final measurement (MedQA test split, n = 1,273)
| Prompt | Unedited Phi-3.5 | Submitted weights | Predicted A / B / C / D (both) |
|---|---|---|---|
| bare | 0.4870 (620) | 0.4870 (620) | 486 / 164 / 359 / 264 |
| chat | 0.5719 (728) | 0.5719 (728) | 252 / 324 / 364 / 333 |

Edited tensors: none. Wrong-to-right 0, right-to-wrong 0. The submitted weights are the original weights.

### The candidate Astra built and then withdrew
- In layers 0 to 7 of 32: the donor's attention weights (`qkv_proj`, `o_proj`) and norm vectors
  (`input_layernorm`, `post_attention_layernorm`) completely, and 256 of the 8,192 MLP units per layer (for unit
  u: rows u and u + 8,192 of `gate_up_proj` and column u of `down_proj`). 46 tensors; the two missing ones
  (`input_layernorm` of layers 6 and 7) are identical in the two checkpoints.
- Size: 320,907,264 weights are in the transfer (8.40 % of the model's 3,821,079,552); 162,464,387 (4.25 %) differ
  from Phi-3.5's after rounding to bfloat16.
- Relative change per tensor (norm of the change divided by the norm of Phi-3.5's tensor):

  | Tensor | Layers | Share of the tensor taken | Relative change |
  |---|---|---|---|
  | `self_attn.o_proj` | 0-7 | all | 3.06 % to 5.31 % |
  | `self_attn.qkv_proj` | 0-7 | all | 0.68 % to 1.25 % |
  | `input_layernorm` | 0-5 | all | 0.001 % to 13.39 % (layer 0) |
  | `post_attention_layernorm` | 0-7 | all | 0.001 % to 1.83 % (layer 0) |
  | `mlp.gate_up_proj` | 0-7 | 256 of 8,192 units (3.1 %) | 0.20 % to 0.33 % (the whole donor tensor: 1.14 % to 1.90 %) |
  | `mlp.down_proj` | 0-7 | 256 of 8,192 units (3.1 %) | 0.24 % to 0.39 % (the whole donor tensor: 1.33 % to 2.20 %) |

- The 256 units per layer carry 3.0 % to 3.3 % of the squared size of the donor's MLP weight change in their
  layer. Astra's notes describe their selection as "label-free from weighted captured activation differences" on
  the discovery questions.
- In MediPhi-Instruct the two norm vectors differ from Phi-3.5's only in the early layers: `input_layernorm` in
  layers 0 to 5 (13.39 % in layer 0, 5.46 % in layer 1, below 1 % after that), `post_attention_layernorm` in
  layers 0 to 10 (1.83 % in layer 0, below 0.02 % after that). The weight matrices differ in every layer
  (`o_proj` 0.89 % to 5.31 %, `qkv_proj` 0.68 % to 1.69 %, MLP 1.14 % to 2.20 %).

Paired results, candidate against the unedited model, same questions (accuracy differences in percentage points):

| Question set | Role | Prompt | Pass | n | Unedited | Candidate | Difference | SE | Wrong to right | Right to wrong | Sign test p | Margin change (SE) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| exploration | used to build and choose | bare | 13 | 256 | 46.09 % | 50.39 % | +4.30 | 2.23 | 22 | 11 | 0.080 | +0.140 (0.106) |
| exploration | used to build and choose | chat | 14 | 256 | 56.25 % | 55.86 % | -0.39 | 1.79 | 10 | 11 | 1.000 | -0.414 (0.281) |
| validation | first use of these questions | bare | 18 | 512 | 49.22 % | 51.17 % | +1.95 | 1.68 | 42 | 32 | 0.295 | +0.132 (0.068) |
| validation | first use of these questions | chat | 19 | 512 | 60.55 % | 60.55 % | 0.00 | 1.00 | 13 | 13 | 1.000 | -0.375 (0.168) |
| confirmation | untouched, rule fixed beforehand | bare | 29 | 1,024 | 51.17 % (524) | 52.93 % (542) | +1.76 | 1.32 | 100 | 82 | 0.208 | +0.123 (0.052) |
| confirmation | untouched, rule fixed beforehand | chat | 30 | 1,024 | 61.13 % (626) | 60.84 % (623) | -0.29 | 0.81 | 33 | 36 | 0.810 | -0.212 (0.126) |

- Astra's retention rule was not met: the bare-prompt gain is 1.33 standard errors (approximate 95 % interval
  -0.82 to +4.34 points) and the chat-prompt difference is negative. Astra removed the edit.
- For scale, MediPhi-Instruct itself on the same 1,024 confirmation questions (EXP-04 files): bare 54.79 % (561),
  chat 60.16 % (616). The candidate's bare-prompt gain is 49 % of the distance from the unedited model (51.17 %)
  to the donor.
- Probability on the four answer letters, bare prompt, confirmation questions: unedited 0.043, candidate 0.134,
  donor 0.413. Under the chat prompt all three are above 0.998.
- Predicted letters A / B / C / D, bare prompt, confirmation questions: unedited 405 / 168 / 243 / 208, candidate
  380 / 187 / 220 / 237, donor 333 / 188 / 213 / 290; correct answers 277 / 262 / 262 / 223.
- Bare-prompt result on the confirmation questions, split by whether the unedited model answers the question
  correctly under the chat prompt (Astra computed the same split on the validation questions before the
  confirmation passes):

  | Unedited model under the chat prompt | n | Unedited, bare | Candidate, bare | Difference | SE | Wrong to right | Right to wrong | Sign test p |
  |---|---|---|---|---|---|---|---|---|
  | right | 626 | 424 | 462 | +6.07 | 1.73 | 79 | 41 | 0.0007 |
  | wrong | 398 | 100 | 80 | -5.03 | 1.96 | 21 | 41 | 0.015 |

  On the 512 validation questions (pass 18, Astra's figures): right n = 310, +4.19 points (30 / 17); wrong
  n = 202, -1.49 points (12 / 15).

### The withdrawn candidate on the test split (after the session)
Candidate against the unedited model, MedQA test split, n = 1,273, paired. One run; the candidate was fixed and
saved before it.

| Prompt | Unedited | Candidate | Difference | SE | Wrong to right | Right to wrong | Sign test p | Margin change (SE) | MediPhi-Instruct itself |
|---|---|---|---|---|---|---|---|---|---|
| bare | 48.70 % (620) | 50.98 % (649) | +2.28 | 1.10 | 113 | 84 | 0.046 | +0.244 (0.046) | 54.60 % (695) |
| chat | 57.19 % (728) | 56.32 % (717) | -0.86 | 0.75 | 40 | 51 | 0.294 | -0.164 (0.105) | 56.48 % (719) |

- Approximate 95 % intervals: bare +0.12 to +4.44 points, chat -2.33 to +0.60 points.
- Astra's retention rule applied to these numbers: the bare-prompt gain is 2.07 standard errors (above 2); the
  chat-prompt difference is negative. The rule is not met.
- The researcher's test of success (bare accuracy up, chat accuracy not down): bare is up by 2.28 points; chat
  is down by 0.86 points (11 answers net), which is 1.15 standard errors.
- The bare-prompt gain is 39 % of the distance from the unedited model to MediPhi-Instruct on these questions.
- Probability on the four answer letters, bare prompt: unedited 0.041, candidate 0.124, donor 0.408.
- Predicted letters A / B / C / D, bare prompt: unedited 486 / 164 / 359 / 264, candidate 475 / 208 / 306 / 284,
  donor 413 / 210 / 315 / 335; correct answers 353 / 309 / 346 / 265.
- Bare-prompt result split by whether the unedited model answers the question correctly under the chat prompt:

  | Unedited model under the chat prompt | n | Unedited, bare | Candidate, bare | Difference | SE | Wrong to right | Right to wrong | Sign test p |
  |---|---|---|---|---|---|---|---|---|
  | right | 728 | 505 | 539 | +4.67 | 1.50 | 77 | 43 | 0.002 |
  | wrong | 545 | 115 | 110 | -0.92 | 1.61 | 36 | 41 | 0.649 |

- The unedited model's letter log-probabilities in this run are identical, on all 1,273 questions and both
  prompts, to those of the harness's final evaluation earlier the same day.
- Confirmation and test questions together (computed by Claude after seeing both; not planned before the run.
  The two sets are disjoint and the candidate was fixed before either was scored), n = 2,297: bare prompt 213
  wrong to right and 166 right to wrong, +2.05 points (SE 0.85, sign test p = 0.018); chat prompt 73 and 87,
  -0.61 points (SE 0.55, sign test p = 0.304).

### Claude's checks of the above
- Recount. All 22 score passes were recounted from the lab's per-question files with the answer keys of this
  repo: every count, difference, standard error and margin agrees with the lab's pass summaries, and the
  confirmation figures agree with Astra's `final_report.json` (accuracy, standard error, flips, margin, letter
  probability, and the split by chat result).
- Split hygiene. The confirmation questions appear in no forward request before pass 29, and the confirmation set
  is disjoint from every other set Astra used. The exploration questions are the first 256 of Astra's 512 "tuning"
  questions; the validation questions are the first 512 of its 1,024 "development check" questions.
- Saved candidate against the checkpoints. No entry outside the transfer differs from Phi-3.5's. Inside it, 45
  entries differ from the donor's by at most 1.2e-10; all others are the donor's exactly. The support and the
  number of changed entries equal the figures in Astra's manifest.
- Saved candidate against the confirmation passes. The lab records the names of a pass's edited tensors, not
  their content. Scoring the saved candidate again on the 1,024 confirmation questions gave 542 correct (bare) and
  623 (chat), with no difference in any letter log-probability of any question from the lab's files of passes 29
  and 30. (The unedited model's log-probabilities differ from the candidate's by 3.2 and 3.7 on average on the
  same questions.)

### Every other transfer Astra scored (all removed by Astra)
Bare prompt unless noted. The tensors are from the lab's record; which heads or units inside them is from
Astra's notes.

| Pass | Transfer | Questions | n | Difference | SE | Wrong to right / right to wrong | Margin change (SE) |
|---|---|---|---|---|---|---|---|
| 4 | Layer 11: the whole MLP | exploration | 256 | +0.78 | 0.78 | 3 / 1 | -0.011 (0.006) |
| 5 | Layer 0: `input_layernorm` | exploration | 256 | +0.78 | 0.78 | 3 / 1 | +0.044 (0.014) |
| 7 | Six attention heads (L7 H1 and H16, L18 H5, H7 and H9, L25 H9) | exploration | 256 | +0.39 | 0.68 | 2 / 1 | -0.003 (0.007) |
| 8 | Layers 0-7: attention weights and `input_layernorm` | exploration | 256 | +3.12 | 2.21 | 20 / 12 | +0.092 (0.101) |
| 9 | Four norm vectors (layer 0 both, layers 1 and 2 `input_layernorm`) | exploration | 256 | +1.17 | 0.87 | 4 / 1 | +0.028 (0.016) |
| 10 | `mlp.down_proj` of layers 0-2 and 4-12 | exploration | 256 | +1.17 | 1.30 | 7 / 4 | -0.000 (0.029) |
| 11 | Layers 0-7: every weight | exploration | 256 | +4.30 | 2.61 | 28 / 17 | +0.261 (0.121) |
| 12 | Layers 0-7: `mlp.gate_up_proj`, and layer 0 `input_layernorm` | exploration | 256 | +0.78 | 0.96 | 4 / 2 | +0.057 (0.026) |
| 15 | Candidate at half strength | exploration | 256 | +0.39 | 1.41 | 7 / 6 | +0.016 (0.057) |
| 16 | Layer 0: every weight | exploration | 256 | +0.78 | 0.96 | 4 / 2 | +0.050 (0.021) |
| 17 | Control: attention and 256 random MLP units in 8 later layers (13, 15, 16, 18, 19, 25, 27, 30) | exploration | 256 | +1.56 | 0.95 | 5 / 1 | +0.036 (0.024) |
| 20 | Control: the candidate with 256 other, random MLP units per layer | exploration | 256 | +2.73 | 2.24 | 20 / 13 | +0.098 (0.102) |
| 21 | Layers 0-7: every weight | validation | 512 | +1.95 | 1.77 | 46 / 36 | +0.216 (0.080) |
| 22 | Candidate at 1.5 times strength | validation | 512 | +4.30 | 1.97 | 62 / 40 | +0.391 (0.099) |
| 23 | Candidate at 1.5 times strength, chat prompt | validation | 512 | -0.98 | 1.28 | 19 / 24 | -1.020 (0.251) |
| 24 | Control: the same 8 later layers as pass 17 | validation | 512 | -0.39 | 0.73 | 6 / 8 | +0.021 (0.017) |

Direct comparisons between two edited models on the same questions (recomputed by Claude):

| Comparison | n | Accuracy difference | SE | Better / worse | Margin difference (SE) |
|---|---|---|---|---|---|
| candidate vs control with random MLP units (13 vs 20) | 256 | +1.56 | 0.78 | 4 / 0 | +0.042 (0.018) |
| candidate vs attention only (13 vs 8) | 256 | +1.17 | 0.67 | 3 / 0 | +0.048 (0.019) |
| every weight of layers 0-7 vs candidate (11 vs 13) | 256 | 0.00 | 1.84 | 11 / 11 | +0.121 (0.034) |
| every weight of layers 0-7 vs candidate (21 vs 18) | 512 | 0.00 | 1.07 | 15 / 15 | +0.084 (0.024) |
| candidate vs control in later layers (18 vs 24) | 512 | +2.34 | 1.68 | 43 / 31 | +0.110 (0.064) |
| candidate at 1.5 times strength vs candidate (22 vs 18) | 512 | +2.34 | 1.54 | 37 / 25 | +0.259 (0.054) |

### Diagnosis and patching numbers
- The two unedited models on Astra's 3,248 development questions (recomputed from the EXP-04 files): bare prompt
  Phi-3.5 47.66 % (1,548), MediPhi-Instruct 54.03 % (1,755); chat prompt 59.39 % (1,929) and 57.14 % (1,856).
  Probability on the four letters under the bare prompt: 0.040 and 0.411. Bare-prompt outcomes: only the donor
  right 475, only the base right 268, both 1,280, neither 1,225.
- Patching on the 64 discovery questions, bare prompt (recomputed from the lab's files; mean margin: unedited
  -0.225, donor +0.377, difference 0.602). Mean change of the margin when one piece of the donor's activity is
  placed into the receiving model:

  | Patched piece | Into unedited Phi-3.5 (passes 3 and 6) | Into the candidate (pass 28) |
  |---|---|---|
  | residual stream after layer 15 | +0.630 (0.331) | +0.202 (0.169) |
  | residual stream after layer 7 | +0.508 (0.280) | +0.080 (0.094) |
  | attention output, layer 18 | +0.433 (0.171) | +0.113 (0.096) |
  | attention output, layer 25 | +0.420 (0.155) | +0.053 (0.067) |
  | MLP output, layer 11 | +0.420 (0.117) | +0.102 (0.062) |
  | head 9 of layer 25 | +0.348 (0.137) | +0.021 (0.058) |
  | head 5 of layer 18 | +0.217 (0.053) | +0.072 (0.038) |
  | head 7 of layer 18 | +0.193 (0.053) | +0.013 (0.020) |
  | head 1 of layer 7 | +0.186 (0.043) | +0.018 (0.014) |

  The candidate's own mean margin on these 64 questions is +0.203 (71 % of the way from the unedited model to the
  donor). The discovery questions were used to choose the candidate's MLP units.
- Astra's own analysis, not rechecked (`work/mechanism_analysis.json`): the change of the residual stream caused
  by the candidate has cosine 0.989 with the donor's change after layer 8.

## Observations
- `[OBSERVATION]` Astra submitted no edit. On the test split the submitted weights score exactly as the unedited
  model under both prompts (0.4870 bare, 0.5719 chat, n = 1,273).
- `[OBSERVATION]` The candidate's bare-prompt gain was +4.30 points (SE 2.23) on the 256 questions used to build
  and choose it, +1.95 (SE 1.68) on 512 questions used for the first time, and +1.76 (SE 1.32) on 1,024 untouched
  questions. None of the three exceeds 2 standard errors.
- `[OBSERVATION]` Under the chat prompt the candidate's accuracy difference is -0.39, 0.00 and -0.29 points on
  the three sets, and the mean margin falls on all three (-0.41, -0.38, -0.21).
- `[OBSERVATION]` On the confirmation questions the candidate's mean bare-prompt margin rises by +0.123 (SE
  0.052, 2.4 standard errors), and the probability on the four letters rises from 0.043 to 0.134 (donor: 0.413).
- `[OBSERVATION]` Transfers chosen from the largest patching effects did not move accuracy: the whole MLP of
  layer 11 (+0.78 points, margin -0.011) and six attention heads (+0.39 points, margin -0.003), against patching
  effects of +0.42 and +0.09 to +0.35 on the margin.
- `[OBSERVATION]` Every transfer with a bare-prompt difference above +2 points on the exploration questions
  contains the donor's attention weights of layers 0 to 7 at full strength (passes 8, 11, 13, 20: +3.12, +4.30,
  +4.30, +2.73). The nine other transfers scored on those questions gave +0.39 to +1.56 points.
- `[OBSERVATION]` Copying every weight of layers 0 to 7 gave the same accuracy as the candidate on the same
  questions (0.00 points difference on 256 and on 512 questions) with a larger margin (+0.121 and +0.084).
- `[OBSERVATION]` Against its controls the candidate has 4 answers better and 0 worse of 256 (+1.56 points, SE
  0.78, sign test p = 0.125) compared with the same transfer with random MLP units, and +2.34 points (SE 1.68, 43
  better and 31 worse of 512) compared with a transfer of the same size in later layers.
- `[OBSERVATION]` The candidate at 1.5 times strength gained +4.30 points (SE 1.97, sign test p = 0.037,
  uncorrected) under the bare prompt on the validation questions and lost 0.98 points (SE 1.28) under the chat
  prompt, with the chat margin falling by 1.02 (SE 0.25). Astra rejected it for the chat result.
- `[OBSERVATION]` On the confirmation questions the candidate's bare-prompt gain is confined to questions the
  unedited model already answers correctly under the chat prompt: +6.07 points (SE 1.73) on those 626 and -5.03
  points (SE 1.96) on the other 398. The validation questions show the same signs (+4.19 and -1.49).
- `[OBSERVATION]` On the test split, scored after the session, the candidate gains +2.28 points (SE 1.10; 113
  fixed, 84 broken, sign test p = 0.046) under the bare prompt and loses 0.86 points (SE 0.75; 40 fixed, 51
  broken) under the chat prompt. Both estimates have the same sign as on the confirmation questions (+1.76 and
  -0.29).
- `[OBSERVATION]` On the test split the bare-prompt gain again sits on the questions the unedited model answers
  correctly under the chat prompt (+4.67 points, SE 1.50, on those 728); on the other 545 the difference is
  -0.92 points (SE 1.61), against -5.03 (SE 1.96) on the confirmation questions.
- `[OBSERVATION]` With the candidate installed, the effect of patching the donor's activity is smaller for all
  14 pieces that were patched into both models (for example head 9 of layer 25: +0.348 to +0.021). Nine of them
  are in the table above.
- `[OBSERVATION]` Astra's final answer states that the evidence favors "prompt adaptation and extraction of
  existing answers, not demonstrated acquisition of medical knowledge". This is Astra's reading, recorded here as
  part of the run's output.

## Negative / null results
- The primary outcome is null: the test-split difference is 0.0 under both prompts, because nothing was
  submitted.
- No transfer tested in 30 passes met Astra's rule on untouched questions (bare-prompt gain above 2 standard
  errors and no fall under the chat prompt).
- The candidate does not meet that rule on the test split either: the bare-prompt gain is above 2 standard
  errors there, and chat-prompt accuracy falls by 0.86 points.
- Copying the weights of the components with the largest activation-patching effects (layer 11 MLP, six heads)
  changed at most 4 of 256 answers.
- The control transfer in later layers changed 6 answers of 256 and 14 of 512.

## Failed / excluded runs
- Two forward requests rejected before any model ran (see Deviations). No Codex reply failed.

## Anomalies
- The unedited model has 524 (bare) and 626 (chat) of the 1,024 confirmation questions right in the lab's passes
  and 522 and 625 in the EXP-04 files (float16 scoring with different batch composition). Each paired comparison
  uses the unedited result from its own pass; the donor's figures are from the EXP-04 files.
- The exploration questions were scored in 14 passes and the validation questions in 6; results on them after the
  first use are not independent of Astra's choices.
- 45 entries of the saved candidate differ from the donor's by at most 1.2e-10 (Astra rebuilt them as base plus
  difference in float32).
- The exploration set has an unedited bare-prompt accuracy of 46.09 %, the confirmation set 51.17 % and the test
  split 48.70 %; the sets differ in difficulty, so only paired differences are comparable across them.
- The lab does not record the content of the edited tensors of a pass. For passes 29 and 30 the second scoring
  closes this; for the earlier passes the description of what was copied inside each tensor rests on Astra's
  notes.

## Scope
One session (`transfer1`), one editor model at effort `xhigh`, one base model and one donor, 30 forward passes,
MedQA USMLE 4-option, single-pass letter scoring under two prompts. Edits were restricted to weights other than
the embeddings, the output layer and the final norm, with no gradients and no simulation of the network in the
kernel. All of Astra's measurements are on practice questions (MedQA dev and training). The candidate was scored
on the test split once, after the session; no control and no other transfer was scored there.

## What this experiment does NOT establish
- That the candidate's bare-prompt gain is settled, in either direction. The two untouched estimates are +1.76
  points (SE 1.32, confirmation) and +2.28 points (SE 1.10, test). The test-split sign test (p = 0.046) is a
  single comparison close to the usual threshold, and the pooled figure was computed after both results were
  known. The candidate does change the model's outputs (margin, letter probability).
- That the chat prompt is unharmed, or harmed. The chat-prompt difference is -0.29 (SE 0.81) and -0.86 (SE 0.75)
  points on the two untouched sets; the experiment cannot distinguish a loss of about one point from none.
- That weights cannot carry the donor's advantage into the base model. Copying every weight gives the donor by
  construction; one session with 30 passes did not find a smaller set that met its own rule.
- That the candidate, or the early layers, hold medical knowledge. The split by chat result and the rise in
  letter probability are descriptive; no measurement here separates medical content from handling a prompt
  without a chat template. EXP-04's control already shows that the two models tie when the same text is wrapped in
  the chat template.
- That the candidate is better than a control of the same size. Against random MLP units it differs by 4 of 256
  answers, on the questions that were used to choose the candidate; against the later-layer control the
  difference is 1.4 standard errors. Neither control was scored on the confirmation questions or the test split,
  so the test-split gain is not shown to be specific to these weights.
- Anything about other donors (`microsoft/MediPhi`), other prompts, or generated answers.
- That the split by chat result is causal. It conditions on an outcome of the same model on the same questions.

## Reproducibility
- Commit: the session started from the working tree committed as `98112a8` (23:36, five minutes after the
  start); see Deviations for the lock change in `69a1948`. `check_session.py` and the exporter are in the commit
  that adds this file.
- Seed: Astra's split seed 815731 (`work/notes.md`). Scoring is deterministic for a fixed set of questions; it
  can differ in the last float16 digits when the batch composition differs.
- Environment: Apple M4 Max, MPS, float16; Codex CLI 0.169 (ChatGPT.app binary).
- Commands:
  `python experiments/exp_05_medical_circuit/run_astra.py --name transfer1 --donor mediphi_instruct --budget 30`
  (rerunning resumes from `raw/transfer1/state.json`; a new `--name` starts a new session);
  `python experiments/exp_05_medical_circuit/check_session.py --name transfer1 --rescore`;
  `python experiments/exp_05_medical_circuit/check_session.py --name transfer1 --test`;
  `python -m wsp.viz_export exp05c`.
- The test-split scoring (`score_test` in `check_session.py`) was written and smoke-tested on the first 32 test
  questions before the run; it was committed together with its result, not before.
- The checks need `raw/transfer1/work/` (local only). Without it, `processed/check_transfer1.json` and
  `check_transfer1_questions.jsonl` hold their output and are enough for the viewer dataset.

## Status
Awaiting researcher review. Not promoted to `research/07_RESULTS_GROUND_TRUTH.md`.
