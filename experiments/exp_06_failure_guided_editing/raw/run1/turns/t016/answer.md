The token check shows that the space-prefixed answer tokens dominate the recorded outputs, so the original attribution directions were relevant; alternate spellings are not driving the observed effect. Strength 3 also produced an approximately threefold letter shift, confirming the dose-response. One interpretation needs checking before proceeding: the returned final hidden state may already include final normalization, which would make my earlier cross-layer compensation estimate misleading. I will verify that measurement convention and finish preparing the larger selection batch without changing the candidate.

```python
# Hypothesis: the final hidden-state export includes normalization, so cross-layer raw-coordinate comparisons require caution.
print("Forward metadata note:", meta1.get("note", "not supplied"))
gamma = final_norm.numpy()
if np.all(np.abs(gamma) > 1e-8):
    export_rms = np.sqrt(np.mean((hidden[32] / gamma[None, :]) ** 2, axis=1))
    print("RMS of exported hidden_32 divided by final-norm weights, quantiles:",
          np.round(np.quantile(export_rms, [0, .25, .5, .75, 1]), 6).tolist())
print("RMS of hidden_31, quantiles:",
      np.round(np.quantile(np.sqrt(np.mean(hidden[31] ** 2, axis=1)), [0, .5, 1]), 4).tolist())
print("Final-norm weight RMS:", round(float(final_norm.square().mean().sqrt()), 4))
with (work / "notes.md").open("a") as f:
    f.write(
        "\nToken audit: space-prefixed single tokens A=319, B=350, C=315, D=360 dominate "
        "all discovery top-1 outputs. Bare-letter variants rarely appear in top 8.\n"
        "Caution: checking whether exported hidden_32 includes final normalization. "
        "Do not interpret its raw coordinate scale as directly comparable to earlier residual states.\n"
    )
print("\nSelection batch, IDs 256 through 511:")
print(json.dumps(selection_large_ids[256:]))
```