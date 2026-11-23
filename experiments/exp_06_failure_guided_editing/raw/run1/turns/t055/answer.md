The candidate was restored successfully, but the tokenizer audit stopped because encoding a leading-space string produced two tokens. That is a tokenizer encoding detail, not evidence that the evaluation protocol is wrong, and I will not change the protocol. For the interpretability analysis, I will identify the raw vocabulary entries directly and check which fixed contrast reproduces the previously used geometry. The upcoming forward passes will still use the tool's unchanged answer scoring.

```python
# Hypothesis: raw vocabulary entries, rather than standalone string encodings, recover the fixed letter-contrast geometry.
raw_letter_ids = {s:{"bare":tok.token_to_id(s),"space":tok.token_to_id("\u2581"+s)} for s in "ABCD"}
print("Raw vocabulary IDs:",raw_letter_ids)
readout_variants = {}
for variant in ("bare","space"):
    idx=[raw_letter_ids[s][variant] for s in "ABCD"]
    assert all(t is not None for t in idx)
    r=head_original[idx,:]*norm_original[None,:]
    readout_variants[variant]=r-r.mean(dim=0,keepdim=True)
readout_variants["average"]=(readout_variants["bare"]+readout_variants["space"])/2
geometry_errors={}
for variant,r in readout_variants.items():
    geometry_errors[variant]=float((r@orth26).norm()/((r@v26).norm()+1e-12))
    print("Geometry",variant,{"column":(r@v26).numpy().round(5).tolist(),
                            "orthogonal":(r@orth26).numpy().round(7).tolist(),
                            "relative_projection":geometry_errors[variant]})
readout_key=min(geometry_errors,key=geometry_errors.get)
R_fixed=readout_variants[readout_key].numpy()
print("Closest matching readout:",readout_key)
del head_original
with (WORK/"notes.md").open("a") as f:
    f.write("\nForward23 opposite orthogonal scale+5, n=128: "+json.dumps(summarize_scores(get_score(23)))+
        "\nRestored locked scale=-3. Opposite control lowers mean margin; accuracy inconclusive. "
        "Tokenizer audit: tok.encode leading-space string emits two tokens. Use raw vocabulary entries for weight-geometry analysis, "
        "not standalone text encodings. Forward scorer remains unchanged. "
        "Late activation diagnostics will use paired fresh passes24/25 and pre-final-norm layer32 exports.\n")
gc.collect()
```
