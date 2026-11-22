```python
import numpy as np
from scipy.linalg import svd, eigvalsh

np.set_printoptions(precision=4, suppress=True, linewidth=190)
d = 384
P = np.eye(d) - np.ones((d, d)) / d
print("L0 raw value singular values")
for h in range(6):
    a = W["W_v.0"][64 * h : 64 * h + 64].astype(float)
    s = svd(a, compute_uv=False)
    print("H", h, s)
print("\nEarly MLP input spectra, including fixed normalization scale and centering")
for l in [0, 1]:
    F = (W[f"W_fc.{l}"].astype(float) * W[f"ln2_w.{l}"]) @ P
    s = svd(F, compute_uv=False)
    print("L", l, s[:90])

Bs = []
g = W["ln1_w.0"].astype(float)
for h in [1, 2, 4, 5]:
    sl = slice(64 * h, 64 * h + 64)
    Bs.append(P @ W["W_o.0"][:, sl].astype(float) @ (W["W_v.0"][sl].astype(float) * g) @ P)
U, S, VT = svd(np.concatenate(Bs, axis=1), full_matrices=False)
C = U[:, :2]
R = P - C @ C.T
print("\nQK matrices on shared early two-dimensional output subspace")
for l in range(1, 4):
    g = W[f"ln1_w.{l}"].astype(float)
    b = W[f"ln1_b.{l}"].astype(float)
    print("L", l)
    for h in range(6):
        sl = slice(64 * h, 64 * h + 64)
        q = W[f"W_q.{l}"][sl].astype(float)
        k = W[f"W_k.{l}"][sl].astype(float)
        A = P @ (q * g).T @ (k * g) @ P / 8
        kb = (q @ b) @ (k * g) @ P / 8
        qb = (k @ b) @ (q * g) @ P / 8
        T = C.T @ A @ C
        rest = R @ A @ R
        print(
            "H",
            h,
            "clock QK",
            T.ravel(),
            "keybias",
            kb @ C,
            "querybias",
            qb @ C,
            "clock QK energy",
            np.linalg.norm(T) ** 2 / np.linalg.norm(A) ** 2,
            "nonclock trace/norm",
            np.trace(rest),
            np.linalg.norm(rest),
        )
```
