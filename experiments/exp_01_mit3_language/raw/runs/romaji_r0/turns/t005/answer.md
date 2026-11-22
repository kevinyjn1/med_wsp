```python
import numpy as np
from scipy.linalg import svd, eigh, polar
from threadpoolctl import threadpool_limits

threadpool_limits(limits=2)
np.set_printoptions(precision=4, suppress=True, linewidth=160)


def norm(a):
    return float(np.linalg.norm(a))


def cos(a, b):
    return float(np.sum(a * b) / (norm(a) * norm(b) + 1e-30))


A, B = [], []
g = np.asarray(W["ln1_w.0"], dtype=float)
for h in range(6):
    sl = slice(h * 64, (h + 1) * 64)
    q, k, v = [np.asarray(W[n + ".0"][sl], dtype=float) for n in ("W_q", "W_k", "W_v")]
    o = np.asarray(W["W_o.0"][:, sl], dtype=float)
    A.append((q * g).T @ (k * g) / 8)
    B.append(o @ (v * g))

# Compare bilinear weight forms in a common weight-derived spectral basis.
# Integer powers below are powers of an orthogonal matrix, not network steps.
u, s, vt = svd(A[3], full_matrices=False, check_finite=False)
metric = ((u * s) @ u.T + (vt.T * s) @ vt) / 2
ev, vec = eigh(metric, check_finite=False)
ix = np.argsort(ev)[::-1]
ev, vec = ev[ix], vec[:, ix]

print("L0 QK ORIENTATION COMPARISON")
for rank in (16, 24, 32, 40):
    t = vec[:, :rank] / np.sqrt(ev[:rank])
    mats = [t.T @ a @ t for a in A]
    rot, _ = polar(mats[3])
    powers = {k: np.linalg.matrix_power(rot, k) for k in range(-6, 9)}
    print("\nrank", rank)
    for h, m in enumerate(mats):
        fits = sorted([(cos(m, p), k) for k, p in powers.items()], reverse=True)
        comm = norm(m @ rot - rot @ m) / (2 * norm(m) + 1e-30)
        print(
            "H" + str(h),
            "top orientation powers/cosines:",
            [(k, round(v, 4)) for v, k in fits[:5]],
            "commutator",
            round(comm, 4),
        )

print("\nSPATIAL CONCENTRATION OF FIRST-LAYER READ DIRECTIONS")
for h in range(6):
    for label, m in (("QK-query", A[h]), ("QK-key", A[h].T), ("OV-read", B[h].T)):
        u, s, vt = svd(m, full_matrices=False, check_finite=False)
        v = u[:, 0]
        ix = np.argsort(-np.abs(v))[:8]
        print(
            "H" + str(h),
            label,
            "participation",
            round(float(1 / np.sum(v**4)), 2),
            "gamma-weighted-mean",
            round(float(np.sum(v * v * g)), 4),
            "largest-coordinates",
            [(int(i), round(float(v[i]), 3)) for i in ix],
        )

print("\nL0H3 OV-READ VS QK-READ LAYERNORM GAIN")
for label, m in (("OV", B[3]), ("QK", A[3])):
    u, s, vt = svd(m, full_matrices=False, check_finite=False)
    for r in (5, 10, 18, 24, 32):
        leverage = np.sum(vt[:r] ** 2, axis=0) / r
        print(
            label,
            r,
            "mean_gain",
            round(float(leverage @ g), 4),
            "axis_participation",
            round(float(1 / np.sum(leverage**2)), 2),
        )

print("\nSIMPLE MLP WEIGHT-PRODUCT SPECTRA")
print("These products do not include ReLU or a LayerNorm evaluation.")
for l in range(4):
    f = np.asarray(W[f"W_fc.{l}"], dtype=float) * np.asarray(W[f"ln2_w.{l}"], dtype=float)
    p = np.asarray(W[f"W_proj.{l}"], dtype=float)
    m = p @ f
    u, s, vt = svd(m, full_matrices=False, check_finite=False)
    print(
        "L",
        l,
        "norm",
        round(norm(m), 4),
        "trace/norm",
        round(float(np.trace(m) / norm(m)), 4),
        "symmetry",
        round(cos(m, m.T), 4),
        "top20_singular",
        np.round(s[:20], 3),
    )
```