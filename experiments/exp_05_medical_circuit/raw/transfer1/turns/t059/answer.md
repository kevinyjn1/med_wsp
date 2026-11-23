The pre-confirmation audit caught a mismatch in the restored MLP weights, so I am stopping before either final score. No confirmation questions have been evaluated or inspected. I will determine whether this is a restoration error or a numerical difference from reconstructing the full-dose edit with subtraction and addition. The candidate must match the saved, previously evaluated snapshot before confirmation proceeds.

```python
# Hypothesis: the audit mismatch comes from numerical reconstruction, not a change in selected components.
import inspect
for fn in [load_original, load_donor, load]:
    try:
        print(fn.__name__, inspect.getsource(fn)[:1800])
    except (OSError, TypeError):
        print(fn.__name__, "source unavailable",
              getattr(getattr(fn, "__code__", None), "co_names", None))

audit_mismatches = []
snapshot_mismatches = []
for name in list(edited()):
    layer = int(name.split(".")[2])
    b = load_original(name).clone()
    d = load_donor(name)
    e = load(name)
    u = torch.tensor(selected_unit_map[str(layer)], dtype=torch.long)
    if ".mlp.gate_up_proj." in name:
        rr = torch.cat([u, u+8192])
        b[rr] = d[rr]
        expected = b
    elif ".mlp.down_proj." in name:
        b[:,u] = d[:,u]
        expected = b
    else:
        expected = d
    count = int(torch.count_nonzero(e.to(torch.bfloat16) != expected.to(torch.bfloat16)))
    if count:
        audit_mismatches.append({"name":name, "different_bf16":count,
                                 "max_float_difference":float((e-expected).abs().max())})
    sp = W / "candidate_sparse_full" / (name + ".npy")
    if sp.exists():
        s = torch.from_numpy(np.load(sp))
        count_s = int(torch.count_nonzero(e.to(torch.bfloat16) != s.to(torch.bfloat16)))
        if count_s:
            snapshot_mismatches.append({"name":name, "different_bf16":count_s,
                                        "max_float_difference":float((e-s).abs().max())})
        del s
    del b,d,e,expected
gc.collect()
print("DONOR-SUPPORT MISMATCHES", audit_mismatches[:12])
print("SAVED-CANDIDATE MISMATCHES", snapshot_mismatches[:12])
print("Total mismatch counts:",
      sum(x["different_bf16"] for x in audit_mismatches),
      sum(x["different_bf16"] for x in snapshot_mismatches))
```
