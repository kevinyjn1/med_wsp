```python
import numpy as np
from scipy.linalg import svd, svdvals, eigh, eigvals
from threadpoolctl import threadpool_limits

threadpool_limits(limits=2)
np.set_printoptions(precision=4, suppress=True, linewidth=180)

g = np.asarray(W["ln1_w.0"], dtype=float)
f = np.asarray(W["W_fc.0"], dtype=float) * np.asarray(W["ln2_w.0"], dtype=float)
A, B = [], []
for h in range(6):
    sl = slice(64 * h, 64 * (h + 1))
    q, k, v = [np.asarray(W[n + ".0"][sl], dtype=float) for n in ("W_q", "W_k", "W_v")]
    o = np.asarray(W["W_o.0"][:, sl], dtype=float)
    A.append((q * g).T @ (k * g) / 8)
    B.append(o @ (v * g))

u, s, vt = svd(A[3], full_matrices=False, check_finite=False)
metric = ((u * s) @ u.T + (vt.T * s) @ vt) / 2
e, v = eigh(metric, check_finite=False)
v = v[:, np.argsort(e)[::-1]]

print("L0 VALUE SPECTRA AFTER REMOVING THE MAIN L0H3 QK SUBSPACE")
for rank in (16, 24, 32, 40):
    p = v[:, :rank]
    print("Removed rank", rank)
    for h in (0, 3):
        m = B[h] - (B[h] @ p) @ p.T
        ss = svdvals(m, check_finite=False)
        print(
            "H" + str(h),
            "first40:",
            np.round(ss[:40], 4),
            "energy_first18/22/26:",
            np.round([np.sum(ss[:r] ** 2) / np.sum(ss**2) for r in (18, 22, 26)], 4),
        )

print("\nCROSS-COVARIANCE OF MLP WEIGHT DIRECTIONS")
print("The two views are F and F@OV_03, projected into an OV-derived subspace.")
print("This is a statistic of neuron weights, not activations or text transition counts.")
u, s, vt = svd(B[3], full_matrices=False, check_finite=False)
for rank in (12, 16, 18, 20, 22, 26):
    r = vt[:rank].T
    x, y = f @ r, (f @ B[3]) @ r
    x -= x.mean(axis=0)
    y -= y.mean(axis=0)
    x /= np.linalg.norm(x)
    y /= np.linalg.norm(y)
    pooled = (x.T @ x + y.T @ y) / 2
    ee, vv = eigh(pooled, check_finite=False)
    t = vv / np.sqrt(np.maximum(ee, 1e-10))
    cross = t.T @ (x.T @ y) @ t
    se = eigh((cross + cross.T) / 2, eigvals_only=True, check_finite=False)
    ce = eigvals(cross, check_finite=False)
    ce = ce[np.argsort(-np.abs(ce))]
    print("rank", rank, "symmetric_eigenvalues", np.round(se, 4))
    print("largest_complex_eigenvalues", np.round(ce[:8], 4))

print("\nROBUSTNESS OF L3H3 MATCHING ALIGNMENT")
for scaled in (False, True):

    def products(l, h):
        sl = slice(64 * h, 64 * (h + 1))
        gg = np.asarray(W[f"ln1_w.{l}"], dtype=float) if scaled else np.ones(384)
        q, k, val = [np.asarray(W[f"{n}.{l}"][sl], dtype=float) for n in ("W_q", "W_k", "W_v")]
        o = np.asarray(W[f"W_o.{l}"][:, sl], dtype=float)
        return (q * gg).T @ (k * gg) / 8, o @ (val * gg)

    _, predecessor = products(0, 3)
    print("LN gain included:", scaled)
    for h in range(6):
        a, bv = products(3, h)
        m = a @ predecessor
        print(
            "L3H" + str(h),
            "direct_trace/norm",
            round(float(np.trace(a) / np.linalg.norm(a)), 4),
            "composed_trace/norm",
            round(float(np.trace(m) / np.linalg.norm(m)), 4),
            "normalized_composition_alignment",
            round(float(np.trace(m) / (np.linalg.norm(a) * np.linalg.norm(predecessor))), 5),
        )
```