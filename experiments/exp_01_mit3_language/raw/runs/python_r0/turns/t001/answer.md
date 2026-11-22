```python
import numpy as np
from scipy.linalg import svdvals
from threadpoolctl import threadpool_limits

threadpool_limits(4)

np.set_printoptions(precision=4, suppress=True, linewidth=180)
print("Keys and shapes:", [(k, v.shape) for k, v in sorted(W.items())])

for l in range(4):
    print("\nLAYER", l)
    for name in ("ln1_w", "ln1_b", "ln2_w", "ln2_b"):
        a = W[f"{name}.{l}"].astype(float)
        print(
            name, "min/mean/max/std/norm", np.array([a.min(), a.mean(), a.max(), a.std(), np.linalg.norm(a)])
        )
    for name in ("W_q", "W_k", "W_v", "W_o", "W_fc", "W_proj"):
        a = W[f"{name}.{l}"].astype(float)
        s = svdvals(a)
        print(
            name,
            "Frob",
            round(np.linalg.norm(a), 4),
            "sv_top12",
            s[:12],
            "sv_quantiles",
            np.quantile(s, [0, 0.25, 0.5, 0.75, 1]),
        )
    g = W[f"ln1_w.{l}"].astype(float)
    b = W[f"ln1_b.{l}"].astype(float)
    for j in range(6):
        sl = slice(64 * j, 64 * (j + 1))
        q = W[f"W_q.{l}"][sl].astype(float)
        k = W[f"W_k.{l}"][sl].astype(float)
        v = W[f"W_v.{l}"][sl].astype(float)
        o = W[f"W_o.{l}"][:, sl].astype(float)
        qg, kg = q * g, k * g
        qg -= qg.mean(axis=1, keepdims=True)
        kg -= kg.mean(axis=1, keepdims=True)
        m = qg.T @ kg / 8
        ov = o @ v
        sm, so = svdvals(m), svdvals(ov)
        eig = np.linalg.eigvalsh((m + m.T) / 2)
        kb = kg.T @ (q @ b) / 8
        qb = qg.T @ (k @ b) / 8
        print(
            f"L{l}H{j}",
            "q/k/v/o_norm",
            np.round([np.linalg.norm(t) for t in (q, k, v, o)], 3),
            "QK_sv",
            sm[:8],
            "QK_sym_extremes",
            eig[[0, 1, 2, -3, -2, -1]],
            "QK_trace",
            round(np.trace(m), 4),
            "QK_asym",
            round(np.linalg.norm(m - m.T) / (2 * np.linalg.norm(m)), 4),
            "bias_key/query_norm",
            np.round([np.linalg.norm(kb), np.linalg.norm(qb)], 4),
            "OV_sv",
            so[:6],
            "OV_trace",
            round(np.trace(ov), 4),
        )
```
