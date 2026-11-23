The finalization block had a syntax error, so none of it ran and the candidate is still installed. I will rerun the corrected block to calculate the pooled statistics and remove the unsupported edit. The decision rule and all measurements remain unchanged. I will explicitly verify that the submitted edit list is empty before finishing.

```python
# Hypothesis: the confirmation result fails the locked retention rule, so the safe submission is the original model.
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
print("All independent questions:",summarize_scores(get_score(26)+final_rows))
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
print("By true answer:",[
    (s,summarize_scores([r for r in final_rows if r["answer"]==s])) for s in "ABCD"])
assert not accepted, "Unexpected acceptance outcome: inspect before finalizing."
for name in list(edited()):
    drop_edit(name)
assert edited()==[]
report={"candidate":json.loads((WORK/"candidate_lock.json").read_text()),
        "selection":summary_strong,"independent_development":summary26,
        "confirmation":final_summary,"letter_shift_diagnostic":bias_diagnostic,
        "retained":False,"submitted_edits":edited()}
(WORK/"final_results.json").write_text(json.dumps(report,indent=2))
final_note="\n".join([
    "",
    "## FINAL DECISION - all 30 forward passes used",
    "Candidate: L26 MLP down-projection neuron 4131, dominant-letter-orthogonal component scaled to -3.",
    "Dose and directional controls established behavioral effects, but not a reliable medical-answering improvement.",
    "Confirmation 1024: "+json.dumps(final_summary),
    "Independent development 256: "+json.dumps(summary26),
    "Letter-shift diagnostic: "+json.dumps(bias_diagnostic),
    "Retention criterion FAILED. All edits dropped; WORK/edited is empty. Submit original weights.",
    "Do not restore exploratory candidate snapshots. Detailed record: final_results.json.",
    ""
])
with (WORK/"notes.md").open("a") as f:
    f.write(final_note)
print("SUBMITTED EDITS:",edited())
print("FILES IN SUBMISSION DIRECTORY:",sorted(p.name for p in (WORK/"edited").iterdir()))
```
