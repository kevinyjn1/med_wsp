```python
import numpy as np
from scipy.linalg import svdvals
from threadpoolctl import threadpool_limits

np.set_printoptions(precision=4, suppress=True, linewidth=160)
with threadpool_limits(limits=2):
    print("Weight keys and shapes:", [(k, v.shape) for k, v in W.items()])
    print("\nLayerNorm parameters")
    for l in range(4):
        for name in ("ln1_w", "ln1_b", "ln2_w", "ln2_b"):
            a = W[f"{name}.{l}"].astype(float)
            ix = np.argsort(np.abs(a))[-6:][::-1]
            print(
                f"{name}.{l}: mean={a.mean():.4f} sd={a.std():.4f} min={a.min():.4f} max={a.max():.4f}; largest={list(zip(ix.tolist(), np.round(a[ix], 4).tolist()))}"
            )
    print("\nMatrix spectra: Frobenius norm, top 8 singular values, stable rank")
    for l in range(4):
        for name in ("W_q", "W_k", "W_v", "W_o", "W_fc", "W_proj"):
            a = W[f"{name}.{l}"].astype(float)
            s = svdvals(a)
            print(
                f"{name}.{l}: fro={np.linalg.norm(a):.3f} top={s[:8]} stable_rank={np.sum(s * s) / s[0] ** 2:.2f}"
            )
    print("\nHead weight products (no network evaluation)")
    print("QK=W_q_head.T@W_k_head/8; OV=W_o_head@W_v_head")
    print("symfrac is ||(A+A.T)/2||_F/||A||_F; trcos is tr(A)/(sqrt(384)||A||_F)")
    for l in range(4):
        for j in range(6):
            sl = slice(64 * j, 64 * (j + 1))
            q = W[f"W_q.{l}"][sl].astype(float)
            k = W[f"W_k.{l}"][sl].astype(float)
            v = W[f"W_v.{l}"][sl].astype(float)
            o = W[f"W_o.{l}"][:, sl].astype(float)
            a, b = q.T @ k / 8, o @ v
            sa, sb = svdvals(a), svdvals(b)
            af, bf = np.linalg.norm(a), np.linalg.norm(b)
            print(
                f"L{l}H{j}: q/k/v/o_norm={[round(np.linalg.norm(z), 3) for z in (q, k, v, o)]}; "
                f"QK fro={af:.3f} top={np.round(sa[:5], 3)} sr={np.sum(sa * sa) / sa[0] ** 2:.2f} symfrac={np.linalg.norm(a + a.T) / (2 * af):.3f} trcos={np.trace(a) / (np.sqrt(384) * af):.3f}; "
                f"OV fro={bf:.3f} top={np.round(sb[:5], 3)} sr={np.sum(sb * sb) / sb[0] ** 2:.2f} symfrac={np.linalg.norm(b + b.T) / (2 * bf):.3f} trcos={np.trace(b) / (np.sqrt(384) * bf):.3f}"
            )
```
