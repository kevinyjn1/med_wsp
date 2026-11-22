```python
import numpy as np
from scipy.linalg import svd, svdvals, eigh
from threadpoolctl import threadpool_limits

threadpool_limits(4)
np.set_printoptions(precision=3, suppress=True, linewidth=160)


def norm(x):
    return np.linalg.norm(x)


def cent(x):
    return x - x.mean(axis=1, keepdims=True)


def basis(x, r):
    return svd(x, full_matrices=False)[0][:, :r]


A, B, V, O = {}, {}, {}, {}
for l in range(4):
    d = W[f"ln1_w.{l}"].astype(float)
    for j in range(6):
        t = (l, j)
        sl = slice(64 * j, 64 * (j + 1))
        q, k, v = [cent(W[f"{n}.{l}"][sl].astype(float) * d) for n in ["W_q", "W_k", "W_v"]]
        o = W[f"W_o.{l}"][:, sl].astype(float)
        A[t] = q.T @ k / 8
        B[t] = o @ v
        V[t] = v
        O[t] = o

print("L0 VALUE SPECTRA")
for j in range(6):
    print(f"H{j} V:", svdvals(V[0, j])[:35])
    print(f"H{j} OV:", svdvals(B[0, j])[:35])

p = basis(np.concatenate([A[0, 1], A[0, 1].T, A[0, 5], A[0, 5].T], axis=1), 26)
c = basis(np.concatenate([V[0, 1].T, V[0, 5].T], axis=1), 27)
print("\nCandidate positional versus value-content subspace overlap:", norm(p.T @ c) ** 2 / 26)
print("Canonical cosines between V1 and V5 top27:", svdvals(basis(V[0, 1].T, 27).T @ basis(V[0, 5].T, 27)))
for j in range(6):
    vv = basis(V[0, j].T, 3)
    print(
        f"H{j} Vtop3 projection fractions: positional={norm(p.T @ vv) ** 2 / 3:.3f}, "
        f"content={norm(c.T @ vv) ** 2 / 3:.3f}; "
        f"QK positional block energy={norm(p.T @ A[0, j] @ p) ** 2 / norm(A[0, j]) ** 2:.3f}, "
        f"content block={norm(c.T @ A[0, j] @ c) ** 2 / norm(A[0, j]) ** 2:.3f}"
    )

print("\nPOSITIONAL-SUBSPACE PROPAGATION")
base = p
for l in range(4):
    if l:
        pieces = [p]
        for m in range(l):
            for j in range(6):
                pieces.append(B[m, j] @ p)
        z = np.concatenate(pieces, axis=1)
        pl = basis(z, 40 if l == 1 else 60)
    else:
        pl = p
    print("Layer", l, "candidate structural dimension", pl.shape[1])
    for j in range(6):
        a = A[l, j]
        lp = norm(pl.T @ a) ** 2 / norm(a) ** 2
        rp = norm(a @ pl) ** 2 / norm(a) ** 2
        pp = norm(pl.T @ a @ pl) ** 2 / norm(a) ** 2
        print(f"L{l}H{j}: Q-side={lp:.3f} K-side={rp:.3f} both={pp:.3f}")

print("\nINDUCTION-LIKE COMPOSITIONS: CONTENT-RESTRICTED TRACE")
for l in range(1, 4):
    for j in range(6):
        vals = []
        for m in range(l):
            for h in range(6):
                z = c.T @ A[l, j] @ B[m, h] @ c
                tr = np.trace(z) / (norm(z) + 1e-30)
                sym = (z + z.T) / 2
                ev = np.linalg.eigvalsh(sym)
                vals.append((tr, f"L{m}H{h}", norm(z), np.sum(ev[ev > 0] ** 2) / (np.sum(ev**2) + 1e-30)))
        vals.sort(reverse=True)
        print(f"L{l}H{j}:", [(tag, round(tr, 3), round(n, 3), round(pos, 3)) for tr, tag, n, pos in vals[:4]])

print("\nL0 MLP INPUT ALIGNMENTS WITH CANDIDATE SUBSPACES")
f = cent(W["W_fc.0"].astype(float) * W["ln2_w.0"])
for tag, z in [("position", p), ("content", c)]:
    energy = np.sum((f @ z) ** 2, axis=1) / np.sum(f * f, axis=1)
    print(tag, "row-energy quantiles:", np.quantile(energy, [0, 0.1, 0.5, 0.9, 0.99, 1]))
```