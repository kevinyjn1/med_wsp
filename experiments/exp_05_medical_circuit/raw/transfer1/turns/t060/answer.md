The audit found only 45 extremely small rounding differences from literal donor values, caused by computing base plus donor-minus-base. Crucially, the current edit matches the saved, previously evaluated candidate exactly after bfloat16 rounding, so this is not a restoration error and no weights need changing. I will use that saved-candidate identity as the checkpoint audit, while retaining the support checks. The sealed confirmation can now proceed with the same candidate that produced the reported development results.

```python
# Hypothesis: the current checkpoint is exactly the tested candidate despite harmless donor-reconstruction rounding.
assert not snapshot_mismatches
assert sum(x["different_bf16"] for x in audit_mismatches) == 45
print("Maximum reconstruction difference:",
      max(x["max_float_difference"] for x in audit_mismatches))
snapshot_files = {p.name for p in (W / "candidate_sparse_full").glob("*.npy")}
assert snapshot_files == {n + ".npy" for n in edited()}
assert len(edited()) == 46
assert all(n.startswith("model.layers.") and int(n.split(".")[2]) < 8 for n in edited())
assert set(confirmation_ids).isdisjoint(seen_score_ids)
assert set(confirmation_ids).isdisjoint(discovery_ids)
with (W / "notes.md").open("a") as f:
    f.write("\nPre-confirmation audit: literal donor equality failed for 45 tiny entries "
            "due to float32 b+(d-b) reconstruction, max discrepancy "
            f"{max(x['max_float_difference'] for x in audit_mismatches):.3g}. "
            "All 46 current edited tensors match the saved, previously tested "
            "candidate_sparse_full checkpoint exactly after bfloat16. No weights changed "
            "during audit. Candidate support remains 8.39834%; actually changed 4.25179%. "
            "Pass28 downstream patch attenuation: L25H9 +0.3477 -> +0.0205; "
            "L18H7 +0.1929 -> +0.0127; L7H1 +0.1860 -> +0.0176; "
            "L11MLP +0.4204 -> +0.1016. Final two passes reserved for sealed n1024 "
            "BARE and CHAT, same unchanged checkpoint. Retain only if BARE gain >2 "
            "paired SE and CHAT accuracy difference >=0; otherwise drop all edits.\n")
print("CONFIRMATION IDS FIRST HALF:")
print(json.dumps(confirmation_ids[:512], separators=(",", ":")))
```
