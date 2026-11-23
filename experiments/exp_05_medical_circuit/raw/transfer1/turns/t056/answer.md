The candidate’s matched traces are now available, including the same discovery questions previously run through both unedited models. I will compare their top continuations, exact probe probabilities, and residual changes before the last causal diagnostic. The key distinction is whether the edit merely shifts fixed letter preferences or also changes how the question controls those preferences. Either finding can explain prompt adaptation, but neither alone establishes a transfer of medical facts.

```python
# Hypothesis: the transferred computation is prompt-sensitive answer extraction with a question-dependent component.
from collections import Counter
probe_donor_results = read_probe_results(26)
probe_edited_results = read_probe_results(27)
probe_comparison = []
for i, (tag, _, ans) in enumerate(probe_specs):
    bs, ds, es = [x[i] for x in [probe_base_results, probe_donor_results, probe_edited_results]]
    entry = {
        "probe": tag,
        "pred_base_donor_edit": [r["pred"] for r in [bs,ds,es]],
        "mass_base_donor_edit": [round(r["letter_mass"],5) for r in [bs,ds,es]],
        "conditional_base": np.round(bs["conditional_ABCD"],3).tolist(),
        "conditional_donor": np.round(ds["conditional_ABCD"],3).tolist(),
        "conditional_edit": np.round(es["conditional_ABCD"],3).tolist()
    }
    probe_comparison.append(entry)
    print(entry)

print("\nQUESTION-DEPENDENT CHANGE AFTER SUBTRACTING MATCHED QUESTIONLESS CHANGE")
for model_tag, rr in [("donor", probe_donor_results), ("edited", probe_edited_results)]:
    for i in [0,1,3,4]:
        blank_i = 2 if i < 3 else 5
        delta_real = np.array(rr[i]["letter_logprobs"]) - np.array(probe_base_results[i]["letter_logprobs"])
        delta_blank = np.array(rr[blank_i]["letter_logprobs"]) - np.array(probe_base_results[blank_i]["letter_logprobs"])
        qdep = delta_real - delta_blank
        qdep -= qdep.mean()
        print(model_tag, probe_specs[i][0], np.round(qdep,3).tolist())

print("\nDISCOVERY TOP NEXT TOKENS")
for pass_num, tag in [(1,"base"),(2,"donor"),(27,"candidate")]:
    top = np.load(W / f"fwd_{pass_num}" / "topk_ids.npy")
    print(tag, Counter(tok.decode([int(i)]) for i in top[:64,0]).most_common(8))

print("\nRESIDUAL TRANSFER ALIGNMENT ON DISCOVERY")
alignment = []
for layer in [1,8,12,18,26,32]:
    b = np.load(W / "fwd_1" / f"hidden_{layer}.npy")[:64].astype(np.float32)
    d = np.load(W / "fwd_2" / f"hidden_{layer}.npy")[:64].astype(np.float32)
    e = np.load(W / "fwd_27" / f"hidden_{layer}.npy")[:64].astype(np.float32)
    db, eb = d-b, e-b
    entry = {"layer":layer,
             "delta_cosine":round(float((db*eb).sum()/np.sqrt((db*db).sum()*(eb*eb).sum())),4),
             "donor_gap_norm_remaining":round(float(np.linalg.norm(d-e)/np.linalg.norm(db)),4)}
    alignment.append(entry)
    print(entry)
(W / "mechanism_analysis.json").write_text(json.dumps({
    "probes":probe_comparison, "activation_alignment":alignment,
    "conditional_extraction":extraction_breakdown}, indent=2))
print("\nPatch record schemas:", list(patch3[0]), list(patch6[0]))
with (W / "notes.md").open("a") as f:
    f.write("\nForward passes used: 27/30. Matched original/donor/edited probe analyses "
            "saved in mechanism_analysis.json. Four simple factual probes have base "
            "accuracy at ceiling, so no medical-knowledge gain can be inferred from them. "
            "Next: donor patches into edited receiver to test whether previously useful "
            "downstream patches are redundant after early transfer; then sealed confirmation.\n")
```
