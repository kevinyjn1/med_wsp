The activation comparison matches the predicted sequence: states are identical before the edit, the immediate dominant-letter change is nearly zero, and substantial letter-contrast changes appear in subsequent layers. This demonstrates downstream mediation of those contrasts, but not a specific improvement in medical reasoning. I will now prepare the independent 256-question development-reserve test with the locked weights. Regardless of that result, I will complete the predeclared 1,024-question confirmation without tuning the edit.

```python
# Hypothesis: the locked direction-specific edit generalizes beyond the questions used to select its strength.
import hashlib
locked_weight = load(name26)
locked_hash = hashlib.sha256(locked_weight.numpy().tobytes()).hexdigest()
assert torch.equal(locked_weight[:,4131],parallel26-3.0*orth26)
assert torch.nonzero((locked_weight!=w26_original).any(dim=0)).flatten().tolist()==[4131]
del locked_weight
independent_ids=splits["development_reserve"][:256]
assert not set(independent_ids)&set(splits["selection"][:512])
assert not set(independent_ids)&set(splits["final_confirmation"])
lock=json.loads((WORK/"candidate_lock.json").read_text())
lock["weight_sha256"]=locked_hash
(WORK/"candidate_lock.json").write_text(json.dumps(lock,indent=2))
with (WORK/"notes.md").open("a") as f:
    f.write("\nCandidate hash before independent testing: "+locked_hash+
        "\nPass26 will use development_reserve[:256], not used for candidate selection. "
        "Passes27-30 will cover all final_confirmation IDs in four fixed256 batches with identical weights. "
        "Retention criterion unchanged; no tuning based on either independent set.\n")
print("INDEPENDENT_REQUEST:",json.dumps({"type":"score","model":"both","questions":independent_ids},separators=(",",":")))
gc.collect()
```
