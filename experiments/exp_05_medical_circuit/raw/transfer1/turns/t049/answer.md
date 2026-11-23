The dose comparison confirms the tradeoff: amplification adds BARE benefit but worsens CHAT margins and produces more CHAT regressions. The full-dose sparse edit is now the sole confirmation candidate; a failed confirmation will mean submitting no edit, not trying another candidate on those questions. I will next repeat the same-size random-location control on the development panel used for the candidate. This tests whether the early location matters, rather than merely whether adding a comparable amount of donor change helps.

```python
# Hypothesis: the selected early prefix is more effective than an equally sized donor transfer at random later locations.
candidate_parameter_count = 8 * (4 * 3072**2 + 3 * 3072 * 256) + 14 * 3072
total_parameter_count = 32 * (4 * 3072**2 + 3 * 3072 * 8192 + 2 * 3072) + 2 * 32064 * 3072 + 3072
print("Selected support:", candidate_parameter_count,
      "of", total_parameter_count,
      "=", round(100 * candidate_parameter_count / total_parameter_count, 5), "%")
candidate_nonzero = 0
candidate_nonzero_bf16 = 0
for name in edited():
    b = load_original(name)
    e = load(name)
    candidate_nonzero += int(torch.count_nonzero(e != b))
    candidate_nonzero_bf16 += int(torch.count_nonzero(e.to(torch.bfloat16) != b.to(torch.bfloat16)))
    del b, e
print("Actually different:", candidate_nonzero,
      "after bfloat16:", candidate_nonzero_bf16,
      "share:", round(100 * candidate_nonzero_bf16 / total_parameter_count, 5), "%")
candidate_manifest = {
    "alpha": 1.0, "unit_map": selected_unit_map,
    "support_count": candidate_parameter_count,
    "changed_bfloat16_count": candidate_nonzero_bf16,
    "total_parameter_count": total_parameter_count,
    "selection_closed_after_pass": 23,
    "confirmation_ids_file": "confirmation_ids.json",
    "decision_rule": "Require positive BARE gain exceeding two paired SEs and no observed CHAT accuracy decline; otherwise submit no edit."
}
(W / "candidate_manifest.json").write_text(json.dumps(candidate_manifest, indent=2))
random_location_map = json.loads((W / "random_location_control.json").read_text())
install_sparse_transfer(1.0, random_location_map)
print("Installed location control; layers:", sorted(map(int, random_location_map)))
print("Next request uses validation_ids, same 512 questions as scores18/19.")
with (W / "notes.md").open("a") as f:
    f.write("\nCandidate manifest saved. Sparse alpha1 support "
            f"{candidate_parameter_count}/{total_parameter_count} "
            f"({100*candidate_parameter_count/total_parameter_count:.5f}%). "
            f"Actually different after bfloat16: {candidate_nonzero_bf16} "
            f"({100*candidate_nonzero_bf16/total_parameter_count:.5f}%). "
            "Confirmation retention rule: BARE gain >2 paired SE and CHAT point estimate >=0. "
            "Current edit temporarily replaced with same-size random later-location control "
            "alpha1 for development replication; restore candidate afterward.\n")
```
