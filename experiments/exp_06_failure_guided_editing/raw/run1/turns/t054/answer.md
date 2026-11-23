The opposite-direction control did not significantly change accuracy, but it worsened the mean margin by 1.053 ± 0.339, unlike the candidate's positive screening margin. This supports a directional effect on answer scores, while leaving the accuracy mechanism unresolved. I will restore the locked candidate and collect matched late-layer activations under the corrected export convention. The aim is to locate where a perturbation initially orthogonal to the chosen letter contrasts becomes a change in answer preference, not to choose further edits.

```python
# Hypothesis: the locked perturbation changes answer preferences through later layers rather than solely through its immediate letter projection.
print("Opposite-direction control:",summarize_scores(get_score(23)))
print("Restored locked scale=-3:",install_orthogonal_scale(-3.0))
assert edited()==[name26]
q_by_id = {}
with (PRACTICE_DIR/"questions.jsonl").open() as f:
    for line in f:
        obj=json.loads(line)
        if obj["id"] not in set(splits["final_confirmation"]):
            q_by_id[obj["id"]]=obj
token_ids = {s:{"bare":tok.encode(s,add_special_tokens=False).ids,
               "space":tok.encode(" "+s,add_special_tokens=False).ids} for s in "ABCD"}
print("Letter token IDs:",token_ids)
norm_original = load_original("model.norm.weight")
head_original = load_original("lm_head.weight")
readout_variants = {}
for variant in ("bare","space"):
    idx=[token_ids[s][variant][0] for s in "ABCD"]
    assert all(len(token_ids[s][variant])==1 for s in "ABCD")
    r=head_original[idx,:]*norm_original[None,:]
    readout_variants[variant]=r-r.mean(dim=0,keepdim=True)
readout_variants["average"]=(readout_variants["bare"]+readout_variants["space"])/2
del head_original
for variant,r in readout_variants.items():
    print("Readout audit",variant,{"column_contrasts":(r@v26).tolist(),
                                "orthogonal_contrasts":(r@orth26).tolist()})
with (WORK/"notes.md").open("a") as f:
    f.write("\nForward23 opposite orthogonal scale+5, n=128: "+json.dumps(summarize_scores(get_score(23)))+
        "\nOpposite control significantly lowers mean margin but accuracy is inconclusive. "
        "Restored locked orthogonal scale=-3. Passes24/25 will collect original/edited late activations under NEW pre-final-norm layer32 convention. "
        "No further candidate tuning planned.\n")
gc.collect()
```
