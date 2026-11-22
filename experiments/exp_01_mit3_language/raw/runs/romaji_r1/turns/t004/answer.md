```python
import numpy as np
from scipy.linalg import svd

np.set_printoptions(precision=4, suppress=True, linewidth=180)
d = 384
P = np.eye(d) - np.ones((d, d)) / d
for l in range(4):
    print("\nLAYER", l)
    for name in ["W_proj", "W_o", "W_fc"]:
        a = W[f"{name}.{l}"].astype(float)
        s = svd(a, compute_uv=False)
        print(name, "first 64 singular values", s[:64])
        ratios = s[:-1] / s[1:]
        ix = np.argsort(-ratios[:80])[:6]
        print("largest early gaps after ranks", [(int(i + 1), round(ratios[i], 3)) for i in ix])

print("\nLayer 0 OV dominant input/output geometry")
Bs = []
As = []
g = W["ln1_w.0"].astype(float)
for h in range(6):
    sl = slice(64 * h, 64 * h + 64)
    B = P @ W["W_o.0"][:, sl].astype(float) @ (W["W_v.0"][sl].astype(float) * g) @ P
    A = P @ (W["W_q.0"][sl].astype(float) * g).T @ (W["W_k.0"][sl].astype(float) * g) @ P / 8
    Bs.append(B)
    As.append(A)
U, S, VT = svd(np.concatenate([Bs[h] for h in [1, 2, 4, 5]], axis=1), full_matrices=False)
clock = U[:, :2]
print("shared early output spectrum", S[:12])
for h in range(6):
    u, s, vt = svd(As[h], full_matrices=False)
    ub, sb, vb = svd(Bs[h], full_matrices=False)
    print(
        "H",
        h,
        "OV energy in shared two-output space",
        np.linalg.norm(clock.T @ Bs[h]) ** 2 / np.linalg.norm(Bs[h]) ** 2,
    )
    print("leading QK query/key cosine", u[:, 0] @ vt[0], "QK leading energy fraction", s[0] ** 2 / (s @ s))
    print("top QK key to top OV input overlaps", vt[:2] @ vb[:3].T)

print("\nLayer 3 output subspace common to projection and attention")
up, sp, vp = svd(P @ W["W_proj.3"].astype(float), full_matrices=False)
for r in [16, 20, 22, 24, 26, 28, 32]:
    basis = up[:, :r]
    print(
        "r",
        r,
        "proj energy",
        np.sum(sp[:r] ** 2) / np.sum(sp**2),
        "head output fractions",
        [
            round(
                np.linalg.norm(basis.T @ W["W_o.3"][:, 64 * h : 64 * h + 64]) ** 2
                / np.linalg.norm(W["W_o.3"][:, 64 * h : 64 * h + 64]) ** 2,
                4,
            )
            for h in range(6)
        ],
    )
```
