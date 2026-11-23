The independent development test did not reproduce the accuracy gain: it corrected one answer and broke three, a change of -0.78 ± 0.78 percentage points. Its margin improvement does not compensate for that result, so the candidate remains unproven. I will now run the reserved confirmation set with the same locked weights and no further tuning. Unless the complete 1,024-question paired result meets the recorded acceptance threshold, I will remove the edit.

```python
# Hypothesis: only the untouched confirmation set can establish whether the selected gain generalizes.
summary26=summarize_scores(get_score(26))
print("Independent development result:",summary26)
assert hashlib.sha256(load(name26).numpy().tobytes()).hexdigest()==locked_hash
final_ids=splits["final_confirmation"]
assert len(final_ids)==len(set(final_ids))==1024
assert not set(final_ids)&set(r["id"] for k in [19,20,21,22,23,26] for r in get_score(k))
with (WORK/"notes.md").open("a") as f:
    f.write("\nForward26, independent development_reserve[:256], locked scale=-3: "+json.dumps(summary26)+
        "\nIndependent accuracy result is negative and inconclusive. No adjustment made. "
        "Proceed to all1024 previously untouched final_confirmation questions with identical locked weights. "
        "Retention requires final paired accuracy gain >2 SE and exact two-sided sign-test p<0.05, else remove edit.\n")
for batch in range(2):
    print("FINAL_REQUEST_"+str(27+batch)+":",json.dumps(
        {"type":"score","model":"both","questions":final_ids[256*batch:256*(batch+1)]},
        separators=(",",":")))
```
