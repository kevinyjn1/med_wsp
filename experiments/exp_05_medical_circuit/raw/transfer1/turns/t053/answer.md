The base probe pass completed successfully and returned exact token probabilities. Before comparing models, I will extract the four-letter distributions and the top unconstrained continuations, keeping questionless prompts separate. I will also summarize the candidate’s development gains according to whether the base already answered correctly under CHAT. Gains concentrated in that group would support better access to existing knowledge rather than newly transferred medical knowledge.

```python
# Hypothesis: the candidate mainly helps express answers the base can already select under CHAT.
def read_probe_results(pass_number):
    folder = W / f"fwd_{pass_number}"
    meta = json.loads((folder / "meta.json").read_text())
    token_lp = np.load(folder / "token_logprobs.npy", mmap_mode="r")
    top_ids = np.load(folder / "topk_ids.npy", mmap_mode="r")
    top_lp = np.load(folder / "topk_logprobs.npy", mmap_mode="r")
    rows_out = []
    for pidx, (tag, prompt_text, answer) in enumerate(probe_specs):
        raw_idx = meta["ids"].index(f"text-{5*pidx}")
        letter_lp = []
        for j in range(4):
            idx = meta["ids"].index(f"text-{5*pidx + 1 + j}")
            letter_lp.append(float(token_lp[idx, meta["lengths"][idx] - 2]))
        letter_lp = np.array(letter_lp)
        letter_mass = float(np.exp(letter_lp).sum())
        rows_out.append({
            "probe": tag, "answer": answer,
            "pred": "ABCD"[int(letter_lp.argmax())],
            "letter_mass": letter_mass,
            "letter_logprobs": letter_lp.tolist(),
            "conditional_ABCD": (np.exp(letter_lp) / letter_mass).tolist(),
            "top3": [(tok.decode([int(t)]), round(float(np.exp(lp)), 5))
                     for t, lp in zip(top_ids[raw_idx, :3], top_lp[raw_idx, :3])]
        })
    return rows_out

probe_base_results = read_probe_results(25)
for r in probe_base_results:
    print(r["probe"], "pred", r["pred"],
          "letter_mass", round(r["letter_mass"], 5),
          "conditional_ABCD", np.round(r["conditional_ABCD"], 3).tolist(),
          "top3", r["top3"])

rb = {r["id"]: r for r in read_score(18)}
rc = {r["id"]: r for r in read_score(19)}
extraction_breakdown = {}
for flag in [True, False]:
    subset = [r for i,r in rb.items() if bool(rc[i]["original"]["correct"]) == flag]
    z = np.array([int(r["edited"]["correct"]) - int(r["original"]["correct"]) for r in subset])
    dm = np.array([r["edited"]["margin"] - r["original"]["margin"] for r in subset])
    extraction_breakdown[str(flag)] = {
        "n": len(z), "rescues": int((z>0).sum()), "regressions": int((z<0).sum()),
        "accuracy_change_pp": round(100*z.mean(), 3),
        "paired_se_pp": round(100*z.std(ddof=1)/np.sqrt(len(z)), 3),
        "margin_change": round(dm.mean(), 4)
    }
print("Candidate effect conditional on original CHAT correctness:", extraction_breakdown)
(W / "probe_original_analysis.json").write_text(json.dumps(probe_base_results, indent=2))
with (W / "notes.md").open("a") as f:
    f.write("\nForward passes used: 25/30. Exact base probe probabilities captured. "
            f"Candidate development effect conditional on original CHAT correctness: {extraction_breakdown}. "
            "This is observational evidence about extraction versus knowledge, not a causal "
            "proof of where medical facts are stored.\n")
```
