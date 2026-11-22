```python
import numpy as np
from scipy.linalg import svd

np.set_printoptions(precision=4, suppress=True, linewidth=180)
d = 384
P = np.eye(d) - np.ones((d, d)) / d
coords = np.array([21, 39, 58, 99, 184])
print("Selected residual coordinates:", coords)
for l in range(4):
    print("L", l)
    for name in ["ln1_w", "ln1_b", "ln2_w", "ln2_b"]:
        print(name, W[f"{name}.{l}"][coords])
g = W["ln1_w.0"].astype(float)
for h in range(6):
    sl = slice(64 * h, 64 * h + 64)
    q = (W["W_q.0"][sl].astype(float) * g) @ P
    k = (W["W_k.0"][sl].astype(float) * g) @ P
    v = (W["W_v.0"][sl].astype(float) * g) @ P
    B = P @ W["W_o.0"][:, sl].astype(float) @ v
    u, s, vt = svd(B, full_matrices=False)
    print("\nL0H", h, "OV input singular-vector concentration")
    for j in range(2):
        ix = np.argsort(-np.abs(vt[j]))[:10]
        print(j, "indices", ix, "coefficients", vt[j, ix], "top10 mass", np.sum(vt[j, ix] ** 2))
    for name, a in [("Q", q), ("K", k), ("V", v)]:
        norms = np.linalg.norm(a, axis=0)
        ix = np.argsort(-norms)[:10]
        print(name, "largest column norms", list(zip(ix, np.round(norms[ix], 4))))
    print("QK selected coordinate block", (q.T @ k / 8)[np.ix_(coords, coords)])
    print("OV selected coordinate block", B[np.ix_(coords, coords)])
```
