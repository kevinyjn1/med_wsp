```python
import numpy as np
from scipy.linalg import svd, svdvals
from threadpoolctl import threadpool_limits

threadpool_limits(4)
np.set_printoptions(precision=4, suppress=True, linewidth=200)


def centered(a):
    return a - a.mean(axis=-1, keepdims=True)


g = W["ln1_w.0"].astype(float)
V = centered(W["W_v.0"].astype(float) * g)
uv, sv, vt = svd(V, full_matrices=False)
print("L0 scaled V singular values:", sv[:80])
T = vt[:28].T
P = np.eye(384) - T @ T.T
Ms, Vs, Os = [], [], []
for j in range(6):
    s = slice(64 * j, 64 * (j + 1))
    q = centered(W["W_q.0"][s].astype(float) * g)
    k = centered(W["W_k.0"][s].astype(float) * g)
    v = centered(W["W_v.0"][s].astype(float) * g)
    o = W["W_o.0"][:, s].astype(float)
    m = q.T @ k / 8
    Ms.append(m)
    Vs.append(v)
    Os.append(o)
    print("\nL0H", j)
    print("q singular values:", svdvals(q)[:40])
    print("k singular values:", svdvals(k)[:40])
    print("v singular values:", svdvals(v)[:40])
    print(
        "QK total energy; token/token, other/other fractions:",
        np.linalg.norm(m) ** 2,
        np.linalg.norm(T.T @ m @ T) ** 2 / np.linalg.norm(m) ** 2,
        np.linalg.norm(P @ m @ P) ** 2 / np.linalg.norm(m) ** 2,
    )
    for label, a in [("full", m), ("outside_V28", P @ m @ P)]:
        u, ss, vh = svd(a, full_matrices=False)
        r = min(40, max(1, int(np.sum(ss > ss[0] * 0.1))))
        overlap = svdvals(u[:, :r].T @ vh[:r].T)
        eig = np.linalg.eigvals(a)
        eig = eig[np.argsort(-np.abs(eig))]
        print(label, "rank_at_10pct", r, "singulars", ss[:32])
        print(label, "left/right overlap", overlap[:r])
        print(label, "eigenvalues_top20", eig[:20])
        print(label, "trace/frob/nuclear", np.trace(a), np.linalg.norm(a), ss.sum())

print("\nPairwise QK Frobenius cosines:")
print(np.array([[np.sum(a * b) / (np.linalg.norm(a) * np.linalg.norm(b)) for b in Ms] for a in Ms]))
print("\nPairwise V input Gram cosines:")
G = [v.T @ v for v in Vs]
print(np.array([[np.sum(a * b) / (np.linalg.norm(a) * np.linalg.norm(b)) for b in G] for a in G]))
print("\nPairwise output subspace overlaps (top 24):")
U = [svd(o, full_matrices=False)[0][:, :24] for o in Os]
print(np.array([[np.linalg.norm(a.T @ b) ** 2 / 24 for b in U] for a in U]))
```
