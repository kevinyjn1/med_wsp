Both matched forward passes completed, including all layers' residual states and MLP activations. I will now inspect the actual preferred next tokens, rather than infer the model's behavior from answer-letter mass alone. I will also summarize where activations diverge, but treat those differences only as descriptive evidence; the next causal experiment will patch attention, MLP, and residual outputs across all 32 layers.

```python
# Hypothesis: donor changes suppress non-answer continuations and alter a localized answer-extraction computation.
from collections import Counter
fwd_base, fwd_donor = work / "fwd_1", work / "fwd_2"
def read_npy(folder, name):
    return np.load(folder / (name + ".npy")).astype(np.float32)

def token_string(t):
    return tok.decode([int(t)], skip_special_tokens=False)

for tag, folder in [("base", fwd_base), ("donor", fwd_donor)]:
    ids = np.load(folder / "topk_ids.npy")
    logp = read_npy(folder, "topk_logprobs")
    common = Counter(token_string(t) for t in ids[:64, 0])
    print(tag, "medical top-1 tokens:", json.dumps(common.most_common(12), ensure_ascii=True))
    for j in [0, 1, 2, 64, 65, 66, 67]:
        pairs = [(token_string(t), round(float(np.exp(p)), 5))
                 for t, p in zip(ids[j, :6], logp[j, :6])]
        print(tag, discovery_ids[j] if j < 64 else "probe_" + str(j - 64),
              json.dumps(pairs, ensure_ascii=True))

activation_summary = []
for layer in range(33):
    b = read_npy(fwd_base, "hidden_" + str(layer))[:64]
    d = read_npy(fwd_donor, "hidden_" + str(layer))[:64]
    delta = d - b
    rel = np.linalg.norm(delta, axis=1) / np.maximum(np.linalg.norm(b, axis=1), 1e-12)
    common_fraction = float(np.sum(delta.mean(0) ** 2) /
                            max(float(np.mean(np.sum(delta ** 2, axis=1))), 1e-20))
    activation_summary.append([layer, round(float(rel.mean()), 4), round(common_fraction, 4)])
print("Residual differences [state index, relative norm, common-direction energy fraction]:")
print(activation_summary)
mlp_drift = []
for layer in range(32):
    b = read_npy(fwd_base, "mlp_act_" + str(layer))[:64]
    d = read_npy(fwd_donor, "mlp_act_" + str(layer))[:64]
    rel = np.linalg.norm(d - b, axis=1) / np.maximum(np.linalg.norm(b, axis=1), 1e-12)
    mlp_drift.append([layer, round(float(rel.mean()), 4)])
print("MLP activation relative differences:", mlp_drift)
del b, d, delta
gc.collect()

with notes_path.open("a") as f:
    f.write("\nForward passes used: 2/30. Matched BARE teacher-forced outputs: fwd_1 base, "
            "fwd_2 donor; 64 discovery questions + 4 nonmedical/empty probes.\n"
            "Residual activation summary: " + json.dumps(activation_summary) + "\n"
            "Next: exhaustive layerwise causal patch screen, not selection by weight magnitude.\n")
```
