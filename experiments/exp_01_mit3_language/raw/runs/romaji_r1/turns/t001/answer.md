```python
import numpy as np
from scipy.linalg import svdvals

np.set_printoptions(precision=4, suppress=True)
print("KEYS", sorted(W))
for l in range(4):
    print("\nLAYER", l)
    for name in ["ln1_w", "ln1_b", "ln2_w", "ln2_b"]:
        a = W[f"{name}.{l}"].astype(float)
        print(name, "min/mean/max/std/norm", np.min(a), np.mean(a), np.max(a), np.std(a), np.linalg.norm(a))
    for name in ["W_q", "W_k", "W_v", "W_o", "W_fc", "W_proj"]:
        a = W[f"{name}.{l}"].astype(float)
        s = svdvals(a)
        print(name, "norm", np.linalg.norm(a), "sv", s[:12], "stable_rank", np.sum(s * s) / (s[0] * s[0]))
    g = W[f"ln1_w.{l}"].astype(float)
    b = W[f"ln1_b.{l}"].astype(float)
    for h in range(6):
        sl = slice(64 * h, 64 * (h + 1))
        q = W[f"W_q.{l}"][sl].astype(float)
        k = W[f"W_k.{l}"][sl].astype(float)
        v = W[f"W_v.{l}"][sl].astype(float)
        o = W[f"W_o.{l}"][:, sl].astype(float)
        A = (q * g).T @ (k * g) / 8
        B = o @ (v * g)
        sa, sb = svdvals(A), svdvals(B)
        print(
            "H",
            h,
            "QKnorm",
            np.linalg.norm(A),
            "QKsv",
            sa[:8],
            "QKsym",
            np.linalg.norm(A + A.T) / (2 * np.linalg.norm(A)),
            "QKtrace",
            np.trace(A),
            "OVnorm",
            np.linalg.norm(B),
            "OVsv",
            sb[:6],
            "OVtrace",
            np.trace(B),
            "key_bias_norm",
            np.linalg.norm((q @ b) @ (k * g)) / 8,
            "query_bias_norm",
            np.linalg.norm((k @ b) @ (q * g)) / 8,
        )
```
