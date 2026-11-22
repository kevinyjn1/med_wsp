```python
import numpy as np
from scipy.linalg import svdvals, eigvalsh
from threadpoolctl import threadpool_limits

np.set_printoptions(precision=4, suppress=True)
threadpool_limits(4)


def norm(x):
    return np.linalg.norm(x)


def effective_rank(s):
    p = s**2 / np.sum(s**2)
    return np.exp(-np.sum(p * np.log(p + 1e-30)))


print("LAYER NORMS")
for l in range(4):
    for name in ["ln1_w", "ln1_b", "ln2_w", "ln2_b"]:
        a = W[f"{name}.{l}"].astype(float)
        print(
            f"{name}.{l}: mean={a.mean():.4f} std={a.std():.4f} "
            f"min={a.min():.4f} max={a.max():.4f} norm={norm(a):.4f}"
        )

print("\nMATRIX STATISTICS")
for l in range(4):
    for name in ["W_q", "W_k", "W_v", "W_o", "W_fc", "W_proj"]:
        a = W[f"{name}.{l}"].astype(float)
        s = svdvals(a)
        print(f"{name}.{l}: norm={norm(a):.3f} top5={s[:5]} erank={effective_rank(s):.1f}")

print("\nHEAD STATISTICS")
print("QK uses Q.T @ K / 8; OV uses O @ V. No activations are evaluated.")
for l in range(4):
    for j in range(6):
        sl = slice(64 * j, 64 * (j + 1))
        q = W[f"W_q.{l}"][sl].astype(float)
        k = W[f"W_k.{l}"][sl].astype(float)
        v = W[f"W_v.{l}"][sl].astype(float)
        o = W[f"W_o.{l}"][:, sl].astype(float)
        a = q.T @ k / 8
        b = o @ v
        sa, sb = svdvals(a)[:64], svdvals(b)[:64]
        ea = eigvalsh((a + a.T) / 2)
        print(
            f"L{l}H{j}: QKnorm={norm(a):.3f} QKtop={sa[:4]} "
            f"QKerank={effective_rank(sa):.1f} "
            f"QKsymcos={np.sum(a * a.T) / (norm(a) ** 2):+.3f} "
            f"QKtrace/norm={np.trace(a) / norm(a):+.3f} "
            f"sym_extrema=({ea[0]:.3f},{ea[-1]:.3f}) "
            f"OVnorm={norm(b):.3f} OVtop={sb[:3]} "
            f"OVerank={effective_rank(sb):.1f} "
            f"OVtrace/norm={np.trace(b) / norm(b):+.3f}"
        )
```