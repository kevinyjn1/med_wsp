The two batches together give 12 corrections and two regressions, about a 1.95-point accuracy gain on 512 development questions. That is substantially stronger than the weaker dose, although the average margin change is close to zero. I will provisionally lock scale -3 as the sole candidate for independent confirmation, then run an equal-sized perturbation in the opposite direction as a causal control. I will also check baseline reproducibility across the restart; small numerical differences should not be mistaken for edit effects.

```python
# Hypothesis: the favorable development result is direction-specific, and an equal-magnitude opposite intervention should not reproduce it.
from scipy.stats import binomtest
rows_strong = get_score(21)+get_score(22)
assert [r["id"] for r in rows_strong] == splits["selection"][:512]
summary_strong = summarize_scores(rows_strong)
summary_strong["sign_test_p"] = float(binomtest(summary_strong["wr"],summary_strong["wr"]+summary_strong["rw"],0.5).pvalue)
print("Strong-dose pooled development:",summary_strong)
old_baseline = {r["id"]:r["original"] for r in get_score(19)}
baseline_pred_changes = sum(r["original"]["pred"]!=old_baseline[r["id"]]["pred"] for r in rows_strong)
baseline_margin_diff = np.array([r["original"]["margin"]-old_baseline[r["id"]]["margin"] for r in rows_strong])
print("Baseline reproducibility:", {"pred_changes":baseline_pred_changes,
    "mean_margin_diff":float(baseline_margin_diff.mean()),"max_abs_margin_diff":float(abs(baseline_margin_diff).max())})
o = np.array([r["original"]["letter_logprobs"] for r in rows_strong])
e = np.array([r["edited"]["letter_logprobs"] for r in rows_strong])
y = np.array(["ABCD".index(r["answer"]) for r in rows_strong])
d = e-o
d -= d.mean(axis=1,keepdims=True)
mu = d.mean(axis=0)
print("Strong-dose letter shift mean / SD:",mu.round(3).tolist(),d.std(axis=0).round(3).tolist())
print("Descriptive accuracy changes, percentage points:",{
    "observed":100*((e.argmax(1)==y).mean()-(o.argmax(1)==y).mean()),
    "mean_letter_shift_only":100*(((o+mu).argmax(1)==y).mean()-(o.argmax(1)==y).mean()),
    "edited_minus_mean_shift":100*(((e-mu).argmax(1)==y).mean()-(o.argmax(1)==y).mean())})
shutil.copytree(WORK/"edited",WORK/"locked_l26_orth_minus3",dirs_exist_ok=True)
lock = {"tensor":name26,"column":4131,"orthogonal_scale":-3.0,
        "selection_passes":[21,22],"final_ids":splits["final_confirmation"],
        "acceptance":"Final 1024 paired accuracy gain > 2 SE and two-sided exact sign-test p < 0.05; otherwise restore original. No tuning on confirmation.",
        "planned_passes":{"23":"opposite-direction control, scale +5, screen128",
                          "24":"original late activations, screen128",
                          "25":"locked edit late activations, screen128",
                          "26":"locked edit independent development_reserve first256",
                          "27-30":"locked edit final_confirmation, four fixed batches of256"}}
(WORK/"candidate_lock.json").write_text(json.dumps(lock,indent=2))
print("Installed opposite-direction control, scale +5:",install_orthogonal_scale(5.0))
with (WORK/"notes.md").open("a") as f:
    f.write("\nForwards21+22, orthogonal scale=-3, pooled selection n=512: "+json.dumps(summary_strong)+
        "\nBaseline vs pre-restart pass19 differs on "+str(baseline_pred_changes)+" predictions. Each test remains paired with its own baseline. "
        "\nCandidate LOCKED: L26:4131 orthogonal scale=-3. Snapshot locked_l26_orth_minus3; rules in candidate_lock.json. "
        "Final1024 acceptance requires positive gain >2 SE and exact two-sided sign-test p<0.05. No tuning on final data. "
        "Pass23 will be a symmetric opposite-direction control (orthogonal scale+5), not a new retention candidate.\n")
```
