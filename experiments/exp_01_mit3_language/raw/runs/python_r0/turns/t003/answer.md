```python
import numpy as np
from scipy.linalg import svd, svdvals
from threadpoolctl import threadpool_limits

threadpool_limits(4)
np.set_printoptions(precision=4, suppress=True, linewidth=200)


def c(a):
    return a - a.mean(axis=-1, keepdims=True)


print("UNSCALED L0 VALUE SPECTRA")
for j in range(6):
    s = slice(64 * j, 64 * j + 64)
    v = c(W["W_v.0"][s].astype(float))
    print(j, svdvals(v)[:45])

vstack = np.vstack([c(W["W_v.0"][64 * j : 64 * j + 64].astype(float)) for j in (1, 5)])
_, ss, vh = svd(vstack, full_matrices=False)
print("Combined H1/H5 V singular values:", ss[:70])
T = vh[:28].T
print("\nL0 input-subspace overlap fractions with combined H1/H5 V top 28")
for j in range(6):
    s = slice(64 * j, 64 * j + 64)
    for name in ("W_q", "W_k", "W_v"):
        a = c(W[f"{name}.0"][s].astype(float))
        print(
            j,
            name,
            "energy",
            np.linalg.norm(a @ T) ** 2 / np.linalg.norm(a) ** 2,
            "top28_overlap",
            np.linalg.norm(svd(a, full_matrices=False)[2][:28] @ T) ** 2 / 28,
        )

# All following objects are weight products, not evaluations on model inputs.
OV = []
for l in range(4):
    g = W[f"ln1_w.{l}"].astype(float)
    for j in range(6):
        s = slice(64 * j, 64 * j + 64)
        OV.append(W[f"W_o.{l}"][:, s].astype(float) @ c(W[f"W_v.{l}"][s].astype(float) * g))
print("\nQK/earlier-OV composition: normalized trace and signed nuclear ratio")
print("Each item: source head, trace(M @ OV)/(||M||F ||OV||F), trace(M @ OV)/nuclear(M @ OV)")
for l in range(1, 4):
    g = W[f"ln1_w.{l}"].astype(float)
    for j in range(6):
        s = slice(64 * j, 64 * j + 64)
        q = c(W[f"W_q.{l}"][s].astype(float) * g)
        k = c(W[f"W_k.{l}"][s].astype(float) * g)
        m = q.T @ k / 8
        scores = []
        for i in range(6 * l):
            a = m @ OV[i]
            trace = np.trace(a)
            score = trace / (np.linalg.norm(m) * np.linalg.norm(OV[i]))
            # Rank is at most 64, so compute the nuclear norm through small factors.
            left = q.T / 8
            right = k @ OV[i]
            _, r1 = np.linalg.qr(left, mode="reduced")
            _, r2 = np.linalg.qr(right.T, mode="reduced")
            nuclear = svdvals(r1 @ r2.T).sum()
            scores.append((i // 6, i % 6, score, trace / nuclear))
        best = sorted(scores, key=lambda z: -abs(z[2]))[:5]
        print(
            f"L{l}H{j}",
            "best",
            [(f"L{a}H{b}", round(s, 4), round(t, 4)) for a, b, s, t in best],
            "L0H1/H5",
            [(round(scores[i][2], 4), round(scores[i][3], 4)) for i in (1, 5)],
        )

print("\nLN1 L0 largest and smallest coordinates")
g = W["ln1_w.0"]
idx = np.argsort(g)
print("largest:", [(int(i), round(float(g[i]), 5)) for i in idx[-35:][::-1]])
print("smallest:", [(int(i), round(float(g[i]), 5)) for i in idx[:20]])
print("gamma quantiles:", np.quantile(g, [0, 0.1, 0.25, 0.5, 0.75, 0.9, 1]))
```
