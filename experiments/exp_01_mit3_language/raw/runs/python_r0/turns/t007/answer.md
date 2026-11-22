```python
import numpy as np
from scipy.linalg import svd, svdvals
from sklearn.decomposition import FastICA
from threadpoolctl import threadpool_limits

threadpool_limits(4)
np.set_printoptions(precision=4, suppress=True, linewidth=200)


def c(a):
    return a - a.mean(-1, keepdims=True)


def norm(a):
    return np.linalg.norm(a)


def cosine(a, b):
    return np.sum(a * b) / (norm(a) * norm(b) + 1e-30)


g0 = W["ln1_w.0"].astype(float)
vv = np.vstack([c(W["W_v.0"][64 * j : 64 * j + 64].astype(float) * g0) for j in (1, 5)])
_, _, vh = svd(vv, full_matrices=False)
T = vh[:28].T
F = c(W["W_fc.0"].astype(float) * W["ln2_w.0"].astype(float))
ica = FastICA(n_components=28, whiten="unit-variance", random_state=13, max_iter=5000, tol=0.0005)
S = ica.fit_transform(F @ T)
mix = ica.mixing_.copy()
sign = np.where(np.mean(S**3, 0) < 0, -1, 1)
S *= sign
mix *= sign
# Read directions are inferred solely by factorizing MLP weight rows.
D = T @ np.linalg.pinv(mix).T
D /= np.linalg.norm(D, axis=0)
print("ICA weight-direction coordinates: per head Q/K/V read strengths (column-normalized)")
read = []
for j in range(6):
    sl = slice(64 * j, 64 * j + 64)
    row = []
    for name in ("W_q", "W_k", "W_v"):
        a = c(W[f"{name}.0"][sl].astype(float) * g0)
        e = np.linalg.norm(a @ D, axis=0)
        row.append(e)
    read.append(row)
for a in range(28):
    print(
        "axis",
        a,
        "Q",
        np.round([read[j][0][a] for j in range(6)], 3),
        "K",
        np.round([read[j][1][a] for j in range(6)], 3),
        "V",
        np.round([read[j][2][a] for j in range(6)], 3),
    )

# Static residual-coordinate QK and OV products.
M = []
B = []
OV = []
for l in range(4):
    g = W[f"ln1_w.{l}"].astype(float)
    b = W[f"ln1_b.{l}"].astype(float)
    for j in range(6):
        sl = slice(64 * j, 64 * j + 64)
        q = c(W[f"W_q.{l}"][sl].astype(float) * g)
        k = c(W[f"W_k.{l}"][sl].astype(float) * g)
        M.append(q.T @ k / 8)
        B.append(k.T @ (W[f"W_q.{l}"][sl].astype(float) @ b) / 8)
        OV.append(W[f"W_o.{l}"][:, sl].astype(float) @ c(W[f"W_v.{l}"][sl].astype(float) * g))
common = np.mean([b / norm(b) for b in B[6:12]], axis=0)
common /= norm(common)
print(
    "\nHead summary: key-bias norm / alignment with shared L1 direction / key and query composition scores through H5"
)
for i in range(24):
    a = T.T @ M[i] @ OV[5] @ T
    aq = T.T @ OV[5].T @ M[i] @ T
    score = np.trace(a) / (svdvals(a).sum() + 1e-30)
    scoreq = np.trace(aq) / (svdvals(aq).sum() + 1e-30)
    es = np.linalg.eigvalsh((a + a.T) / 2)
    print(
        f"L{i // 6}H{i % 6}",
        np.array(
            [
                norm(B[i]),
                B[i] @ common / (norm(B[i]) + 1e-30),
                score,
                scoreq,
                np.sum(es[es > 0]) / max(np.sum(abs(es)), 1e-30),
            ]
        ),
    )

# Identify MLP0 rows dominated by the positional QK reference space.
_, _, pv = svd(
    np.vstack(
        [
            c(W[f"{name}.0"][64 * j : 64 * j + 64].astype(float) * g0)
            for j in (1, 5)
            for name in ("W_q", "W_k")
        ]
    ),
    full_matrices=False,
)
P = pv[:32].T
frac = np.sum((F @ P) ** 2, axis=1) / np.sum(F**2, axis=1)
ids = np.argsort(frac)[-24:][::-1]
print("\nMLP0 rows most aligned with positional weight subspace:")
for n in ids:
    print(
        int(n),
        "fraction",
        round(frac[n], 4),
        "fc_norm",
        round(norm(F[n]), 4),
        "proj_norm",
        round(norm(W["W_proj.0"][:, n]), 4),
    )
for count in (8, 24, 64):
    ids = np.argsort(frac)[-count:]
    u, _, _ = svd(W["W_proj.0"][:, ids].astype(float), full_matrices=False)
    pp = u[:, : min(count, 24)]
    print("Positional-neuron output reference count", count)
    print(
        np.array(
            [
                [
                    norm(pp.T @ M[i] @ pp) ** 2 / norm(M[i]) ** 2,
                    norm(pp.T @ B[i]) ** 2 / (norm(B[i]) ** 2 + 1e-30),
                ]
                for i in range(6, 24)
            ]
        )
    )
```
