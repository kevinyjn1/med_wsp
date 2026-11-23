**Final submission: unedited base.** I removed all edits and verified that `WORK/edited` is empty. **0% of weights are changed.** No candidate earned retention under the predefined confirmation rule.

### Independent confirmation

The frozen candidate was tested on 1,024 questions excluded from selection:

- **BARE:** 51.17% → 52.93%, **+1.76 ± 1.32 percentage points** (paired SE). There were 100 rescues and 82 regressions; the 95% interval includes zero.
- **CHAT:** 61.13% → 60.84%, **-0.29 ± 0.81 points**. This does not establish harm, but it does not meet the no-observed-decline guardrail.

### What was localized and tested

The strongest candidate transferred donor changes in **layers 0–7**: attention QKV/O projections, changed normalization vectors, and **256 selected MLP units per layer**, including their gate/up rows and down-projection columns. Its support covered 8.40% of parameter positions; **4.25% actually changed** after checkpoint rounding. Excluded tensors remained untouched.

The investigation progressed from output diagnosis through activation patching, direct weight transfers, sparse decomposition, dose tests, and random controls:

- Helpful downstream activation patches did **not** imply sufficient local weight changes: transferring the selected heads or layer-11 MLP alone gave little benefit.
- The sparse early transfer closely reproduced the donor’s early residual change, with cosine similarity **0.989 at layer 8**.
- Downstream patch benefits then shrank substantially, supporting an upstream-source interpretation.
- A same-size random-location control was weaker, but its direct accuracy contrast with the candidate remained inconclusive.
- Amplifying the transfer improved development BARE performance but worsened CHAT. Copying complete early layers did not improve accuracy over the sparse candidate.

### What the computation appears to do

The evidence favors **prompt adaptation and extraction of existing answers**, not demonstrated acquisition of medical knowledge.

The candidate increased confirmation-set letter probability mass from **0.043 to 0.134**, with similar increases on questionless and nonmedical probes. Fixed letter preferences explained some, but not all, of its behavior; increased total letter mass alone cannot change the four-letter argmax.

Question-dependent effects were substantial: BARE accuracy improved **6.07 points** where the base already answered correctly under CHAT, but declined **5.03 points** where it did not. This supports better expression of existing knowledge, although it is not a causal decomposition of medical knowledge.

**Conclusion:** the experiments identified a plausible distributed prompt-handling mechanism, but did not establish a sufficiently reliable, CHAT-safe weight transfer. Findings and statistics are saved in `WORK/notes.md` and `WORK/final_report.json`.