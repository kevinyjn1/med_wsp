```python
import numpy as np
from scipy.linalg import svdvals, eigvals
from threadpoolctl import threadpool_limits

np.set_printoptions(precision=4, suppress=True, linewidth=160)
threadpool_limits(limits=2)


def norm(x):
    return float(np.linalg.norm(x))


def cosine(a, b):
    return float(np.sum(a * b) / (norm(a) * norm(b) + 1e-30))


def spectrum_summary(a):
    s = svdvals(a, check_finite=False)
    e = s * s
    return norm(a), e.sum() / (e[0] + 1e-30), e[:4].sum() / (e.sum() + 1e-30)


print("AVAILABLE WEIGHTS")
for l in range(4):
    print("Layer", l)
    for name in ("ln1_w", "ln1_b", "ln2_w", "ln2_b"):
        a = np.asarray(W[f"{name}.{l}"], dtype=np.float64)
        print(name, "min/mean/max/std/norm:", np.round([a.min(), a.mean(), a.max(), a.std(), norm(a)], 5))
    for name in ("W_q", "W_k", "W_v", "W_o", "W_fc", "W_proj"):
        a = np.asarray(W[f"{name}.{l}"], dtype=np.float64)
        fn, sr, top4 = spectrum_summary(a)
        print(name, a.shape, "Frob/stable_rank/top4_energy:", np.round([fn, sr, top4], 5))

# These are products of weights, not evaluations on inputs.
A, B, labels = [], [], []
print("\nHEAD WEIGHT PRODUCTS")
print("QK = diag(ln1_w) Q.T K diag(ln1_w) / sqrt(64)")
print("OV = O V diag(ln1_w)")
print(
    "Columns: head | QK_norm stable_rank top4_energy symmetry_cos trace/norm "
    "| OV_norm stable_rank top4_energy symmetry_cos trace/norm "
    "| fixed_query_key_vector_norm/QK_norm"
)
for l in range(4):
    g = np.asarray(W[f"ln1_w.{l}"], dtype=np.float64)
    b = np.asarray(W[f"ln1_b.{l}"], dtype=np.float64)
    for h in range(6):
        s = slice(64 * h, 64 * (h + 1))
        q = np.asarray(W[f"W_q.{l}"][s], dtype=np.float64)
        k = np.asarray(W[f"W_k.{l}"][s], dtype=np.float64)
        v = np.asarray(W[f"W_v.{l}"][s], dtype=np.float64)
        o = np.asarray(W[f"W_o.{l}"][:, s], dtype=np.float64)
        a = (q * g).T @ (k * g) / 8.0
        bv = o @ (v * g)
        c = (k * g).T @ (q @ b) / 8.0
        A.append(a)
        B.append(bv)
        labels.append(f"L{l}H{h}")
        an, ar, ae = spectrum_summary(a)
        bn, br, be = spectrum_summary(bv)
        vals = [
            an,
            ar,
            ae,
            cosine(a, a.T),
            np.trace(a) / an,
            bn,
            br,
            be,
            cosine(bv, bv.T),
            np.trace(bv) / bn,
            norm(c) / an,
        ]
        print(labels[-1], " ".join(f"{x:.5f}" for x in vals))

print("\nEARLIER-HEAD OV TO LATER-HEAD QK TRACE ALIGNMENT")
print(
    "For each later head: strongest signed trace(QK_later @ OV_earlier) / "
    "(||QK_later||F ||OV_earlier||F). A positive value can support a matching circuit."
)
for j in range(6, 24):
    pairs = []
    for i in range((j // 6) * 6):
        val = float(np.trace(A[j] @ B[i]) / (norm(A[j]) * norm(B[i])))
        pairs.append((val, labels[i]))
    pairs.sort(reverse=True)
    print(
        labels[j],
        "top:",
        [(lab, round(val, 5)) for val, lab in pairs[:5]],
        "bottom:",
        [(lab, round(val, 5)) for val, lab in pairs[-2:]],
    )

print("\nSTRONGEST CROSS-HEAD QK AND OV SIMILARITIES")
for kind, mats in (("QK", A), ("OV", B)):
    pairs = []
    for j in range(24):
        for i in range(j):
            pairs.append((cosine(mats[i], mats[j]), labels[i], labels[j]))
    pairs.sort(reverse=True)
    print(kind, "positive:", [(a, b, round(v, 5)) for v, a, b in pairs[:12]])
    print(kind, "negative:", [(a, b, round(v, 5)) for v, a, b in pairs[-6:]])
```