On 512 questions, the orthogonal reversal corrected seven answers and broke two: a gain of 0.98 ± 0.58 percentage points, with a positive mean-margin change. The additional questions therefore support the direction of the initial result, but the evidence is still insufficient for retention. I will preserve this candidate and test a larger perturbation of exactly the same component to check dose-response. Before doing so, I will separate the observed score changes into their average letter shift and question-dependent remainder; this is a descriptive control, not proof of improved reasoning.

```python
# Hypothesis: strengthening the same orthogonal intervention produces a coherent dose-response beyond a constant answer-letter shift.
import shutil


def get_score(k):
    return [json.loads(s) for s in (Path(WORK) / f"score_{k}.jsonl").read_text().splitlines()]


def summarize_scores(rows):
    a = np.array([int(r["original"]["correct"]) for r in rows])
    b = np.array([int(r["edited"]["correct"]) for r in rows])
    d = b - a
    m = np.array([r["edited"]["margin"] - r["original"]["margin"] for r in rows])
    return dict(
        n=len(rows),
        original=float(a.mean()),
        edited=float(b.mean()),
        delta_pp=float(100 * d.mean()),
        se_pp=float(100 * d.std(ddof=1) / np.sqrt(len(d))),
        wr=int((d == 1).sum()),
        rw=int((d == -1).sum()),
        margin=float(m.mean()),
        margin_se=float(m.std(ddof=1) / np.sqrt(len(m))),
    )


rows19 = get_score(19)
o19 = np.array([r["original"]["letter_logprobs"] for r in rows19])
e19 = np.array([r["edited"]["letter_logprobs"] for r in rows19])
d19 = e19 - o19
d19 -= d19.mean(axis=1, keepdims=True)
mean19 = d19.mean(axis=0)
y19 = np.array(["ABCD".index(r["answer"]) for r in rows19])
print("Full sample:", summarize_scores(rows19))
print("Outside initial screen:", summarize_scores(rows19[128:]))
print("Centered letter-shift mean / SD:", mean19.round(3).tolist(), d19.std(axis=0).round(3).tolist())
print(
    "Descriptive accuracies:",
    {
        "original": float((o19.argmax(1) == y19).mean()),
        "edited": float((e19.argmax(1) == y19).mean()),
        "mean_shift_only": float(((o19 + mean19).argmax(1) == y19).mean()),
        "edited_minus_mean_shift": float(((e19 - mean19).argmax(1) == y19).mean()),
    },
)
print("By true letter:", [(s, summarize_scores([r for r in rows19 if r["answer"] == s])) for s in "ABCD"])
shutil.copytree(Path(WORK) / "edited", Path(WORK) / "candidate_l26_orth_minus1", dirs_exist_ok=True)
w26_original = load_original("model.layers.26.mlp.down_proj.weight")
v26 = w26_original[:, 4131].clone()
v26_parallel = letter_basis @ (letter_basis.T @ v26)
v26_orthogonal = v26 - v26_parallel


def install_orthogonal_scale(scale):
    for name in list(edited()):
        drop_edit(name)
    w = w26_original.clone()
    w[:, 4131] = v26_parallel + scale * v26_orthogonal
    save_edit("model.layers.26.mlp.down_proj.weight", w)
    return float(torch.linalg.vector_norm(w - w26_original) / torch.linalg.vector_norm(w26_original))


print("Installed orthogonal scale=-3; relative tensor change:", install_orthogonal_scale(-3.0))
with (Path(WORK) / "notes.md").open("a") as f:
    f.write(
        "\nForward 19 orthogonal scale=-1, n=512: "
        + json.dumps(summarize_scores(rows19))
        + "\nOutside initial 128: "
        + json.dumps(summarize_scores(rows19[128:]))
        + "\nCandidate saved to candidate_l26_orth_minus1. Next dose: orthogonal scale=-3, "
        "letter component unchanged. No candidate yet independently confirmed.\n"
    )
```
