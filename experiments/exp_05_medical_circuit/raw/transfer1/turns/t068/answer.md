The CHAT confirmation was 61.13% for the base and 60.84% for the candidate, a change of -0.29 ± 0.81 percentage points. This does not establish harm, but it also fails the precommitted no-observed-decline guardrail, while BARE already failed the accuracy-evidence threshold. I will therefore remove all edits rather than submit an insufficiently supported transfer. I will save the final statistics and mechanistic findings separately, with the active submission verified as the unedited base.

```python
# Hypothesis: neither confirmation result justifies retaining the candidate under the precommitted rule.
for name in list(edited()):
    drop_edit(name)
assert len(edited()) == 0
assert not list((W / "edited").glob("*.npy"))

def final_score_summary(pass_number):
    rows = read_score(pass_number)
    assert len(rows) == 1024
    assert {r["id"] for r in rows} == set(confirmation_ids)
    b = np.array([r["original"]["correct"] for r in rows], dtype=float)
    e = np.array([r["edited"]["correct"] for r in rows], dtype=float)
    z = e-b
    dm = np.array([r["edited"]["margin"]-r["original"]["margin"] for r in rows])
    se = float(z.std(ddof=1)/np.sqrt(len(z)))
    wins, losses = int((z>0).sum()), int((z<0).sum())
    return {
        "n":len(rows), "original_accuracy":float(b.mean()), "candidate_accuracy":float(e.mean()),
        "difference_pp":float(100*z.mean()), "paired_se_pp":100*se,
        "approx_95pct_ci_pp":[float(100*(z.mean()-1.96*se)),float(100*(z.mean()+1.96*se))],
        "rescues":wins, "regressions":losses,
        "sign_test_p":float(binomtest(wins,wins+losses).pvalue),
        "margin_difference":float(dm.mean()),
        "margin_se":float(dm.std(ddof=1)/np.sqrt(len(dm))),
        "original_letter_mass":float(np.exp(score_lp_matrix(rows,"original")).sum(axis=1).mean()),
        "candidate_letter_mass":float(np.exp(score_lp_matrix(rows,"edited")).sum(axis=1).mean())
    }

final_bare = final_score_summary(29)
final_chat = final_score_summary(30)
assert final_bare["difference_pp"] <= 2*final_bare["paired_se_pp"]
assert final_chat["difference_pp"] < 0

fbare = {r["id"]:r for r in read_score(29)}
fchat = {r["id"]:r for r in read_score(30)}
final_conditional = {}
for flag in [True,False]:
    rr = [r for i,r in fbare.items() if bool(fchat[i]["original"]["correct"]) == flag]
    zz = np.array([int(r["edited"]["correct"])-int(r["original"]["correct"]) for r in rr])
    final_conditional[str(flag)] = {
        "n":len(rr), "rescues":int((zz>0).sum()), "regressions":int((zz<0).sum()),
        "difference_pp":float(100*zz.mean()),
        "paired_se_pp":float(100*zz.std(ddof=1)/np.sqrt(len(zz)))
    }

confirmation_rows = read_score(29)
fb = score_lp_matrix(confirmation_rows,"original")
fe = score_lp_matrix(confirmation_rows,"edited")
fy = np.array(["ABCD".index(r["answer"]) for r in confirmation_rows])
final_offset_diagnostic = {
    "offset_source":"separate exploration score13, no confirmation fitting",
    "base_plus_fixed_offset_accuracy":float(((fb+fixed_offset).argmax(axis=1)==fy).mean()),
    "candidate_minus_fixed_offset_accuracy":float(((fe-fixed_offset).argmax(axis=1)==fy).mean()),
    "caution":"Descriptive and nonadditive; not proof of medical-knowledge transfer."
}
final_report = {
    "submission":"unedited base", "edited_tensors":list(edited()),
    "changed_parameter_count":0, "changed_parameter_share":0.0,
    "forward_passes_used":30,
    "rejected_candidate":candidate_manifest,
    "confirmation_bare":final_bare, "confirmation_chat":final_chat,
    "conditional_on_original_chat_correct":final_conditional,
    "fixed_offset_diagnostic":final_offset_diagnostic,
    "interpretation":"Evidence for distributed early prompt adaptation and question-dependent answer extraction; no demonstrated new medical knowledge and no adequately confirmed successful edit."
}
(W / "final_report.json").write_text(json.dumps(final_report,indent=2))
with (W / "notes.md").open("a") as f:
    f.write("\nFINAL: 30/30 forward passes used. ALL ACTIVE EDITS REMOVED AND VERIFIED. "
            "Submission is the unedited base, 0 changed parameters (0%). "
            "Candidate failed precommitted confirmation criterion; no fallback selection. "
            "\nBARE: " + json.dumps(final_bare) +
            "\nCHAT: " + json.dumps(final_chat) +
            "\nConditional extraction diagnostic: " + json.dumps(final_conditional) +
            "\nFixed-offset diagnostic: " + json.dumps(final_offset_diagnostic) +
            "\nFull report: final_report.json. Rejected candidate snapshot retained only "
            "for reproducibility in candidate_sparse_full, not in active edited directory.\n")
print("BARE:",json.dumps(final_bare))
print("CHAT:",json.dumps(final_chat))
print("CONTENT/EXTRACTION:",json.dumps(final_conditional))
print("FIXED OFFSET:",json.dumps(final_offset_diagnostic))
print("FINAL ACTIVE EDITS:",edited())
print("ACTIVE EDIT DIRECTORY:",[p.name for p in (W / "edited").iterdir()])
```
