All 30 passes are complete. Across the four confirmation batches, the candidate corrected 19 answers and broke 12, a net gain of only 0.68 percentage points on 1,024 questions. That is too small relative to its paired uncertainty to meet the predeclared retention rule, despite the promising selection result and positive margin changes. I will compute the exact pooled statistics, record the letter-shift analysis, and remove the edit so the submitted weights remain original.

```python
# Hypothesis: the confirmation gain is insufficiently reliable to justify retaining a weight change.
final_rows=[r for k in range(27,31) for r in get_score(k)]
assert [r["id"] for r in final_rows]==final_ids
assert len({r["id"] for r in final_rows})==1024
assert hashlib.sha256(load(name26).numpy().tobytes()).hexdigest()==locked_hash
final_summary=summarize_scores(final_rows)
final_summary["sign_test_p"]=float(binomtest(
    final_summary["wr"],final_summary["wr"]+final_summary["rw"],0.5).pvalue)
final_summary["approx_95ci_pp"]=[
    final_summary["delta_pp"]-1.96*final_summary["se_pp"],
    final_summary["delta_pp"]+1.96*final_summary["se_pp"]]
accepted=(final_summary["delta_pp"]>2*final_summary["se_pp"]
          and final_summary["sign_test_p"]<0.05)
print("FINAL CONFIRMATION:",json.dumps(final_summary,indent=2))
print("Independent development plus confirmation:",summarize_scores(get_score(26)+final_rows))
od=np.array([r["original"]["letter_logprobs"] for r in rows_strong])
ed=np.array([r["edited"]["letter_logprobs"] for r in rows_strong])
dd=ed-od
development_mean_shift=(dd-dd.mean(axis=1,keepdims=True)).mean(axis=0)
of=np.array([r["original"]["letter_logprobs"] for r in final_rows])
ef=np.array([r["edited"]["letter_logprobs"] for r in final_rows])
yf=np.array(["ABCD".index(r["answer"]) for r in final_rows])
base_acc=(of.argmax(1)==yf).mean()
bias_diagnostic={
    "development_fixed_shift_ABCD":development_mean_shift.tolist(),
    "observed_final_delta_pp":float(100*((ef.argmax(1)==yf).mean()-base_acc)),
    "constant_shift_only_delta_pp":float(100*(((of+development_mean_shift).argmax(1)==yf).mean()-base_acc)),
    "edited_minus_constant_shift_delta_pp":float(100*(((ef-development_mean_shift).argmax(1)==yf).mean()-base_acc)),
    "caution":"Descriptive score decomposition, not additive causal attribution or a new model evaluation."}
print("LETTER-SHIFT DIAGNOSTIC:",json.dumps(bias_diagnostic,indent=2))
print("Final results by true answer:",[
    (s,summarize_scores([r for r in final_rows if r["answer"]==s])) for s in "ABCD"])
if not accepted:
    for name in list(edited()):
        drop_edit(name)
assert not accepted and edited()==[], "Unexpected acceptance outcome: inspect before finalizing."
report={"candidate":json.loads((WORK/"candidate_lock.json").read_text()),
        "selection":summary_strong,"independent_development":summary26,
        "confirmation":final_summary,"letter_shift_diagnostic":bias_diagnostic,
        "retained":False,"submitted_edits":edited()}
(WORK/"final_results.json").write_text(json.dumps(report,indent=2))
with (WORK/"notes.md").open("a") as f:
    f.write("\n## FINAL DECISION - all30 forward passes used\n"+
        "Candidate: L26 MLP down-projection neuron4131, dominant-letter-orthogonal component scaled to-3; all other components original. "
        "Selected from dose and directional controls; late activation effects causally observed but medical reasoning benefit not established.\n"+
        "Confirmation1024: "+json.dumps(final_summary)+"\n"+
        "Independent development256: "+json.dumps(summary26)+"\n"+
        "Letter-shift diagnostic: "+json.dumps(bias_diagnostic)+"\"+
        "Retention criterion FAILED. All edits dropped; WORK/edited is empty. Submit original weights. "
        "Do not restore exploratory candidate snapshots. Detailed record: final_results.json.\n")
print("SUBMITTED EDITS:",edited())
```
