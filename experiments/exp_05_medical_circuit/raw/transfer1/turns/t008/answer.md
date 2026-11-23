The patch file contains per-question margins, which lets me correct descriptive averages for the discovery panel's unequal sampling. I will rank attention and MLP interventions separately from residual replacements, since only the former narrow the search to individual blocks. A useful candidate should improve more than just the selected donor-rescue cases and should survive a direct weight-transfer test on other questions.

```python
# Hypothesis: a small subset of attention or MLP blocks produces robust gains beyond selected rescue cases.
import pandas as pd
population_weights = {g: len(ix) / len(dev_ids) for g, ix in groups.items()}
panel_group_indices = {
    g: np.array([j for j, r in enumerate(patch3) if discovery_group[str(r["id"])] == g])
    for g in groups
}
patch_stats = []
for p in patch3[0]["patched"]:
    delta = np.array([r["patched"][p] - r["base"] for r in patch3])
    weighted_mean = sum(population_weights[g] * delta[ix].mean()
                        for g, ix in panel_group_indices.items())
    weighted_var = sum(population_weights[g] ** 2 * delta[ix].var(ddof=1) / len(ix)
                       for g, ix in panel_group_indices.items())
    patch_stats.append({
        "patch": p, "mean": delta.mean(), "se": delta.std(ddof=1) / 8,
        "weighted_mean": weighted_mean, "weighted_se": weighted_var ** .5,
        **{g: delta[ix].mean() for g, ix in panel_group_indices.items()}
    })
patch_df = pd.DataFrame(patch_stats)
components_df = patch_df[~patch_df.patch.str.endswith("resid")]
print("Component screen, sorted by population-reweighted margin change:")
print(components_df.sort_values("weighted_mean", ascending=False).round(4).to_string(index=False))
print("Early residual progression:")
print(patch_df[patch_df.patch.isin(["L%d.resid" % l for l in range(14)])]
      [["patch", "mean", "weighted_mean", "weighted_se"]].round(4).to_string(index=False))
patch_df.to_csv(work / "patch3_analysis.csv", index=False)
with notes_path.open("a") as f:
    f.write("\nTop population-reweighted component patches:\n" +
            components_df.sort_values("weighted_mean", ascending=False).head(12)
            .round(4).to_string(index=False) + "\n")
```
