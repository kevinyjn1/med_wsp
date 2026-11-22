```python
import numpy as np
from scipy.linalg import svd, eigvals

np.set_printoptions(precision=4, suppress=True, linewidth=180)


def topbasis(a, r):
    return svd(a, full_matrices=False)[2][:r].T


g = W["ln1_w.0"].astype(float)
Q = [W["W_q.0"][64 * h : 64 * h + 64].astype(float) * g for h in range(6)]
K = [W["W_k.0"][64 * h : 64 * h + 64].astype(float) * g for h in range(6)]
V = [W["W_v.0"][64 * h : 64 * h + 64].astype(float) * g for h in range(6)]
O = [W["W_o.0"][:, 64 * h : 64 * h + 64].astype(float) for h in range(6)]
Vall = np.concatenate(V)
_, vs, vt = svd(Vall, full_matrices=False)
print("L0 combined value singular values:", vs[:80])
charbasis = vt[:28].T
pbase = topbasis(np.concatenate([Q[3], K[3]]), 64)
print("\nL0 head subspace diagnostics")
for h in range(6):
    A = Q[h].T @ K[h] / 8
    B = O[h] @ V[h]
    ev = eigvals(A)
    ev = ev[np.argsort(-np.abs(ev))][:48]
    print("\nH", h)
    print("Q singular values", svd(Q[h], compute_uv=False)[:20])
    print("K singular values", svd(K[h], compute_uv=False)[:20])
    print("V singular values", svd(V[h], compute_uv=False)[:20])
    print("QK eigenvalues", ev)
    print(
        "fraction squared norm in combined V top28: Q/K",
        np.linalg.norm(Q[h] @ charbasis) ** 2 / np.linalg.norm(Q[h]) ** 2,
        np.linalg.norm(K[h] @ charbasis) ** 2 / np.linalg.norm(K[h]) ** 2,
    )
    print(
        "fraction squared norm in H3 QK top64: Q/K/V",
        *[np.linalg.norm(a @ pbase) ** 2 / np.linalg.norm(a) ** 2 for a in [Q[h], K[h], V[h]]],
    )
    print("OV stable rank", np.linalg.norm(B) ** 2 / np.linalg.norm(B, 2) ** 2)

print("\nL0 pairwise normalized Frobenius inner products: QK")
AA = [q.T @ k for q, k in zip(Q, K)]
print(np.array([[np.sum(a * b) / np.linalg.norm(a) / np.linalg.norm(b) for b in AA] for a in AA]))
print("\nLayer-to-layer key-bias direction cosine similarities, and headwise bias norms")
for l in range(4):
    g = W[f"ln1_w.{l}"].astype(float)
    b = W[f"ln1_b.{l}"].astype(float)
    dirs = []
    for h in range(6):
        sl = slice(h * 64, (h + 1) * 64)
        q = W[f"W_q.{l}"][sl].astype(float)
        k = W[f"W_k.{l}"][sl].astype(float)
        dirs.append((q @ b) @ (k * g) / 8)
    dirs = np.array(dirs)
    print("L", l, "norms", np.linalg.norm(dirs, axis=1))
    dirs = dirs / np.linalg.norm(dirs, axis=1, keepdims=True)
    print(dirs @ dirs.T)
```
