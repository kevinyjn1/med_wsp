```python
import numpy as np
from scipy.linalg import svd, svdvals
from threadpoolctl import threadpool_limits

threadpool_limits(limits=2)
np.set_printoptions(precision=4, suppress=True, linewidth=180)


def norm(a):
    return float(np.linalg.norm(a))


def cos(a, b):
    return float(np.sum(a * b) / (norm(a) * norm(b) + 1e-30))


g = np.asarray(W["ln1_w.0"], dtype=float)
g2 = np.asarray(W["ln2_w.0"], dtype=float)
As, Bs = [], []
for h in range(6):
    sl = slice(h * 64, (h + 1) * 64)
    q, k, v = [np.asarray(W[name + ".0"][sl], dtype=float) for name in ("W_q", "W_k", "W_v")]
    o = np.asarray(W["W_o.0"][:, sl], dtype=float)
    a, b = (q * g).T @ (k * g) / 8, o @ (v * g)
    As.append(a)
    Bs.append(b)
    print("L0H" + str(h))
    print("QK singular values:", svdvals(a, check_finite=False)[:64])
    print("OV singular values:", svdvals(b, check_finite=False)[:64])
    print("V diag(g) singular values:", svdvals(v * g, check_finite=False))

print("\nL0 EFFECTIVE MLP INPUT SINGULAR VALUES 1..100")
f = np.asarray(W["W_fc.0"], dtype=float) * g2
print(svdvals(f, check_finite=False)[:100])

print("\nSHARED LARGE RESIDUAL COORDINATES")
ix = [21, 39, 58, 99]
for l in range(4):
    print("Layer", l)
    for name in ("ln1_w", "ln1_b", "ln2_w", "ln2_b"):
        print(name, np.asarray(W[f"{name}.{l}"])[ix])

print("\nL0 DOMINANT QUERY/KEY/OV-READ DIRECTIONS")
dirs, names = [], []
for h in range(6):
    u, s, vt = svd(As[h], full_matrices=False, check_finite=False)
    ub, sb, vtb = svd(Bs[h], full_matrices=False, check_finite=False)
    for name, vec in ((f"H{h}Q", u[:, 0]), (f"H{h}K", vt[0]), (f"H{h}V", vtb[0])):
        dirs.append(vec)
        names.append(name)
print("Order:", names)
print(np.array([[cos(a, b) for b in dirs] for a in dirs]))

print("\nFIRST-LAYER SUBSPACE OVERLAPS")
print("Fraction of each QK/OV matrix energy lying in the indicated row or column subspace.")
u, s, vt = svd(Bs[3], full_matrices=False, check_finite=False)
for rank in (16, 20, 22, 24, 28, 32):
    read = vt[:rank].T
    write = u[:, :rank]
    print("L0H3 OV rank", rank)
    for h in range(6):
        vals = [
            norm(read.T @ As[h]) ** 2 / norm(As[h]) ** 2,
            norm(As[h] @ read) ** 2 / norm(As[h]) ** 2,
            norm(Bs[h] @ read) ** 2 / norm(Bs[h]) ** 2,
            norm(write.T @ Bs[h]) ** 2 / norm(Bs[h]) ** 2,
        ]
        print("H" + str(h), np.round(vals, 4))
```