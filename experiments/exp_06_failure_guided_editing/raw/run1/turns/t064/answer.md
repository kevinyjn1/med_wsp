The second confirmation batch had three corrections and four regressions, so the first batch's gain has not repeated consistently. I will complete the remaining 512 questions without changing the candidate. Before those runs, I will verify the weight fingerprint and retrieve the remaining fixed IDs. No retention decision will be made from the partial confirmation results.

```python
# Hypothesis: completing the prespecified confirmation sample will distinguish a reproducible benefit from selection noise.
assert hashlib.sha256(load(name26).numpy().tobytes()).hexdigest()==locked_hash
assert [r["id"] for r in get_score(27)+get_score(28)]==final_ids[:512]
with (WORK/"notes.md").open("a") as f:
    f.write("\nForward27 final batch1: "+json.dumps(summarize_scores(get_score(27)))+
        "\nForward28 final batch2: "+json.dumps(summarize_scores(get_score(28)))+
        "\nCandidate fingerprint unchanged. Complete remaining512 final questions before applying acceptance rule.\n")
for batch in range(2,4):
    print("FINAL_REQUEST_"+str(27+batch)+":",json.dumps(
        {"type":"score","model":"both","questions":final_ids[256*batch:256*(batch+1)]},
        separators=(",",":")))
```
