```python
import numpy as np
from scipy.linalg import svdvals, eigvals
from threadpoolctl import threadpool_limits

threadpool_limits(limits=2)
np.set_printoptions(precision=4, suppress=True, linewidth=180)

Bs, Cs, gs = {}, {}, {}
for l in range(4):
    w = W[f"ln1_w.{l}"].astype(float)
    b = W[f"ln1_b.{l}"].astype(float)
    for j in range(6):
        s = slice(64 * j, 64 * (j + 1))
        q0, k0, v0 = [W[f"W_{a}.{l}"][s].astype(float) for a in ("q", "k", "v")]
        q, k, v = [a * w for a in (q0, k0, v0)]
        q, k, v = [a - a.mean(1, keepdims=True) for a in (q, k, v)]
        B = q.T @ k / 8
        C = W[f"W_o.{l}"][:, s].astype(float) @ v
        g = (q0 @ b) @ k / 8
        Bs[l, j], Cs[l, j], gs[l, j] = B, C, g
        raw = q0.T @ k0 / 8
        ev = eigvals(B)
        ev = ev[np.argsort(-np.abs(ev))]
        print(
            f"L{l}H{j}",
            "raw_QK_norm",
            round(np.linalg.norm(raw), 4),
            "q_bias_norm",
            round(np.linalg.norm(q0 @ b), 4),
            "k_bias_norm",
            round(np.linalg.norm(k0 @ b), 4),
            "key_linear_norm",
            round(np.linalg.norm(g), 4),
            "QK_eigen_top",
            np.round(ev[:8], 4),
        )

print("\nWithin-layer QK Frobenius cosines")
for l in range(4):
    a = np.stack([Bs[l, j].ravel() for j in range(6)])
    a /= np.linalg.norm(a, axis=1, keepdims=True)
    print("Layer", l, "\n", a @ a.T)
    a = np.stack([gs[l, j] for j in range(6)])
    a /= np.linalg.norm(a, axis=1, keepdims=True)
    print("Key-linear direction cosines\n", a @ a.T)

print("\nEarlier OV -> later QK composition")
print("For each later head: earlier head, trace(B C)/[||B||F ||C||F], trace(B C)/||B C||*, transpose cosine")
for l in range(1, 4):
    for j in range(6):
        B = Bs[l, j]
        rows = []
        for a in range(l):
            for h in range(6):
                C = Cs[a, h]
                T = B @ C
                norm = np.linalg.norm(T)
                tr = np.trace(T)
                align = tr / (np.linalg.norm(B) * np.linalg.norm(C))
                nuclear = svdvals(T).sum()
                symcos = np.sum(T * T.T) / (norm * norm)
                rows.append((align, f"L{a}H{h}", tr / nuclear, symcos))
        rows.sort(reverse=True)
        print(f"L{l}H{j}", [(r[1], *np.round([r[0], r[2], r[3]], 4)) for r in rows[:5]])

print("\nL0H2 singular spectrum")
u, s, vt = np.linalg.svd(Bs[0, 2], full_matrices=False)
print(s[:40])
print("Left/right singular-subspace overlap, top 24:\n", np.round(svdvals(u[:, :24].T @ vt[:24].T), 4))
print("\nLayerNorm largest and smallest weight coordinates")
for l in range(4):
    w = W[f"ln1_w.{l}"]
    ix = np.argsort(w)
    print(
        l,
        "largest",
        [(int(i), round(float(w[i]), 4)) for i in ix[-12:]],
        "smallest",
        [(int(i), round(float(w[i]), 4)) for i in ix[:12]],
    )
```
