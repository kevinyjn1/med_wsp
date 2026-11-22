```python
import numpy as np
from scipy.linalg import svdvals
from threadpoolctl import threadpool_limits

np.set_printoptions(precision=4, suppress=True, linewidth=160)
threadpool_limits(limits=2)


def stats(a):
    return (float(np.linalg.norm(a)), float(np.mean(a)), float(np.std(a)), float(np.min(a)), float(np.max(a)))


print("Available weights:", sorted(W))
for l in range(4):
    print(f"\nLAYER {l}")
    for name in ("ln1_w", "ln1_b", "ln2_w", "ln2_b"):
        a = W[f"{name}.{l}"].astype(np.float64)
        print(name, "norm/mean/std/min/max:", np.array(stats(a)))
    for name in ("W_q", "W_k", "W_v", "W_o", "W_fc", "W_proj"):
        a = W[f"{name}.{l}"].astype(np.float64)
        s = svdvals(a)
        print(
            name,
            "Frobenius:",
            round(np.linalg.norm(a), 4),
            "singular values:",
            np.round(s[:8], 4),
            "stable rank:",
            round(np.sum(s * s) / (s[0] * s[0]), 2),
        )
    w = W[f"ln1_w.{l}"].astype(np.float64)
    for j in range(6):
        sl = slice(64 * j, 64 * (j + 1))
        q = W[f"W_q.{l}"][sl].astype(np.float64)
        k = W[f"W_k.{l}"][sl].astype(np.float64)
        v = W[f"W_v.{l}"][sl].astype(np.float64)
        o = W[f"W_o.{l}"][:, sl].astype(np.float64)
        q = q * w
        k = k * w
        v = v * w
        q -= q.mean(axis=1, keepdims=True)
        k -= k.mean(axis=1, keepdims=True)
        v -= v.mean(axis=1, keepdims=True)
        B = q.T @ k / 8.0
        C = o @ v
        sb, sc = svdvals(B), svdvals(C)
        bn, cn = np.linalg.norm(B), np.linalg.norm(C)
        symcos = np.sum(B * B.T) / (bn * bn)
        print(
            f"L{l}H{j}",
            "QK_norm",
            round(bn, 4),
            "QK_top",
            np.round(sb[:6], 4),
            "QK_stablerank",
            round(bn * bn / (sb[0] * sb[0]), 2),
            "QK_transpose_cos",
            round(symcos, 4),
            "QK_trace",
            round(np.trace(B), 4),
            "OV_norm",
            round(cn, 4),
            "OV_top",
            np.round(sc[:6], 4),
            "OV_trace",
            round(np.trace(C), 4),
        )
```
