```python
import numpy as np
from scipy.linalg import svd, svdvals
from threadpoolctl import threadpool_limits

threadpool_limits(limits=2)
np.set_printoptions(precision=4, suppress=True, linewidth=180)


def centered_scaled(a, w):
    a = a.astype(float) * w.astype(float)
    return a - a.mean(axis=1, keepdims=True)


B, C = {}, {}
for l in range(4):
    w = W[f"ln1_w.{l}"]
    for j in range(6):
        sl = slice(64 * j, 64 * j + 64)
        q, k, v = [centered_scaled(W[f"W_{s}.{l}"][sl], w) for s in ("q", "k", "v")]
        B[l, j] = q.T @ k / 8
        C[l, j] = W[f"W_o.{l}"][:, sl].astype(float) @ v

print("Final MLP projection singular values, first 64:")
print(svdvals(W["W_proj.3"].astype(float))[:64])
print("First-layer value singular spectra:")
for j in range(6):
    v = centered_scaled(W["W_v.0"][j * 64 : (j + 1) * 64], W["ln1_w.0"])
    print(j, svdvals(v)[:35])

u, s, vt = svd(B[0, 2], full_matrices=False)
P = u[:, :22] @ u[:, :22].T
print("\nFraction of QK energy in L0H2's top-22 left singular subspace:")
for l in range(4):
    for j in range(6):
        a = B[l, j]
        n = np.linalg.norm(a)
        print(
            f"L{l}H{j}",
            "query",
            round(np.linalg.norm(P @ a) ** 2 / n**2, 4),
            "key",
            round(np.linalg.norm(a @ P) ** 2 / n**2, 4),
            "both",
            round(np.linalg.norm(P @ a @ P) ** 2 / n**2, 4),
        )

print("\nL0 OV leading left-vector overlaps (absolute):")
left = []
for j in range(6):
    u, s, vt = svd(C[0, j], full_matrices=False)
    left.append(u[:, :3])
print(np.abs(np.stack([a[:, 0] for a in left]) @ np.stack([a[:, 0] for a in left]).T))
for j in range(6):
    print(j, "overlap of top-three output subspace with H0:", svdvals(left[0].T @ left[j]))

M = np.concatenate([C[0, j] for j in (0, 1, 4, 5)], axis=1)
u, s, vt = svd(M, full_matrices=False)
print("\nCombined broad-candidate OV singular values:", s[:16])
R = u[:, :3]
print("Leading output-direction coordinates:")
for i in range(3):
    ix = np.argsort(-np.abs(R[:, i]))[:10]
    print(i, [(int(k), round(float(R[k, i]), 4)) for k in ix])

print("\nQK projections onto this common top-three output subspace:")
for l in range(1, 4):
    w = W[f"ln1_w.{l}"].astype(float)
    b = W[f"ln1_b.{l}"].astype(float)
    for j in range(6):
        a = B[l, j]
        sl = slice(64 * j, 64 * j + 64)
        q0 = W[f"W_q.{l}"][sl].astype(float)
        k = centered_scaled(W[f"W_k.{l}"][sl], w)
        g = (q0 @ b) @ k / 8
        print(
            f"L{l}H{j}",
            "energy",
            round(np.linalg.norm(R.T @ a @ R) ** 2 / np.linalg.norm(a) ** 2, 4),
            "matrix",
            np.round(R.T @ a @ R, 3).ravel(),
            "linear",
            np.round(g @ R, 3),
        )

print("\nFirst-layer MLP encoder/decoder spectra:")
for name in ("W_fc.0", "W_proj.0"):
    print(name, svdvals(W[name].astype(float))[:45])
```
