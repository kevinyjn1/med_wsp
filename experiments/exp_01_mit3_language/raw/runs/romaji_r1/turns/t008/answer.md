```python
import numpy as np
from scipy.linalg import svd, svdvals

np.set_printoptions(precision=4, suppress=True, linewidth=180)
d = 384
P = np.eye(d) - np.ones((d, d)) / d
A = {}
B = {}
for l in range(4):
    g = W[f"ln1_w.{l}"].astype(float)
    for h in range(6):
        sl = slice(64 * h, 64 * h + 64)
        q = (W[f"W_q.{l}"][sl].astype(float) * g) @ P
        k = (W[f"W_k.{l}"][sl].astype(float) * g) @ P
        A[l, h] = q.T @ k / 8
        B[l, h] = P @ W[f"W_o.{l}"][:, sl].astype(float) @ (W[f"W_v.{l}"][sl].astype(float) * g) @ P

print("Key versus query composition through L0H3")
for l in range(1, 4):
    for h in range(6):
        key = A[l, h] @ B[0, 3]
        query = B[0, 3].T @ A[l, h]
        print(
            "L%dH%d" % (l, h),
            "key trace/nuclear",
            np.trace(key) / svdvals(key).sum(),
            "query trace/nuclear",
            np.trace(query) / svdvals(query).sum(),
        )

print("\nCross-correlations between static current and transported MLP input-weight factors")
F = (W["W_fc.0"].astype(float) * W["ln2_w.0"]) @ P
F = F - F.mean(0)
G = F @ B[0, 3]
u, s, vt = svd(G, full_matrices=False)
for r in [12, 16, 20, 24, 28]:
    T = vt[:r].T
    X = F @ T
    Y = G @ T
    H = (X.T @ X + Y.T @ Y) / 2
    ev, U = np.linalg.eigh(H)
    inv = U @ np.diag(1 / np.sqrt(np.maximum(ev, 1e-10))) @ U.T
    C = inv @ (X.T @ Y + Y.T @ X) / 2 @ inv
    vals = np.linalg.eigvalsh(C)
    print("r", r, "symmetric whitened weight-correlation eigenvalues", vals)

print("\nInduction key-composition null from residual-coordinate permutations")
rng = np.random.default_rng(123)
for lh in [(1, 1), (2, 5), (3, 0), (3, 3)]:
    a = A[lh]
    b = B[0, 3]
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    obs = np.trace(a @ b) / denom
    null = []
    for _ in range(1000):
        p = rng.permutation(d)
        bp = b[p][:, p]
        null.append(np.sum(a * bp.T) / denom)
    null = np.array(null)
    print(
        lh,
        "observed",
        obs,
        "null mean/std/max",
        null.mean(),
        null.std(),
        null.max(),
        "z",
        (obs - null.mean()) / null.std(),
    )

U, S, VT = svd(np.concatenate([B[0, h] for h in [1, 2, 4, 5]], axis=1), full_matrices=False)
print("\nConcentration of shared output directions in individual residual coordinates")
for j in range(3):
    ix = np.argsort(-np.abs(U[:, j]))[:10]
    print(
        "direction",
        j,
        "top coordinates",
        ix,
        "coefficients",
        U[ix, j],
        "top2/top10 squared mass",
        np.sum(U[ix[:2], j] ** 2),
        np.sum(U[ix, j] ** 2),
    )
```
