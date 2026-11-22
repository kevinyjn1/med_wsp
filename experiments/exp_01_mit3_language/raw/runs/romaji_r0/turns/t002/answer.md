```python
import numpy as np
from scipy.linalg import svd, eigvals
from threadpoolctl import threadpool_limits

threadpool_limits(limits=2)
np.set_printoptions(precision=4, suppress=True, linewidth=180)


def norm(a):
    return float(np.linalg.norm(a))


def cos(a, b):
    return float(np.sum(a * b) / (norm(a) * norm(b) + 1e-30))


A, B, C, labels = [], [], [], []
for l in range(4):
    g = np.asarray(W[f"ln1_w.{l}"], dtype=float)
    b = np.asarray(W[f"ln1_b.{l}"], dtype=float)
    for h in range(6):
        s = slice(h * 64, (h + 1) * 64)
        q, k, v = [np.asarray(W[f"{name}.{l}"][s], dtype=float) for name in ("W_q", "W_k", "W_v")]
        o = np.asarray(W[f"W_o.{l}"][:, s], dtype=float)
        A.append((q * g).T @ (k * g) / 8)
        B.append(o @ (v * g))
        C.append((k * g).T @ (q @ b) / 8)
        labels.append(f"L{l}H{h}")

print("FIRST-LAYER QK EIGENVALUES, SORTED BY MAGNITUDE")
for j in range(6):
    e = eigvals(A[j], check_finite=False)
    e = e[np.argsort(-np.abs(e))]
    print(labels[j], e[:16])

print("\nFIRST-LAYER OV SINGULAR VALUES")
for j in range(6):
    u, s, vt = svd(B[j], full_matrices=False, check_finite=False)
    print(labels[j], s[:16])

# A shared residual direction, defined entirely by an OV weight product.
u, s, vt = svd(B[5], full_matrices=False, check_finite=False)
r = u[:, 0]
t = vt[0]
if r[np.argmax(np.abs(r))] < 0:
    r, t = -r, -t

print("\nL0H5 DOMINANT OV LEFT AND RIGHT DIRECTIONS")
for name, vec in (("left", r), ("right", t)):
    ix = np.argsort(-np.abs(vec))[:20]
    print(
        name,
        "largest coordinates:",
        [(int(i), round(float(vec[i]), 5)) for i in ix],
        "participation_ratio:",
        round(float(1 / np.sum(vec**4)), 3),
    )

print("\nSHARED DIRECTION ALIGNMENT WITH LAYERNORM WEIGHTS AND BIASES")
for l in range(4):
    for name in ("ln1_w", "ln1_b", "ln2_w", "ln2_b"):
        v = np.asarray(W[f"{name}.{l}"], dtype=float)
        print(l, name, "cos(left), cos(right):", round(cos(r, v), 5), round(cos(t, v), 5))

print("\nHEAD DEPENDENCE ON THE SHARED OV OUTPUT DIRECTION")
print(
    "Columns: head, fraction_QK_energy_on_query_direction, "
    "fraction_QK_energy_on_key_direction, fixed_key_vector_cos_direction, "
    "OV_output_energy_on_direction, OV_input_energy_on_direction"
)
for j in range(24):
    vals = [
        norm(r @ A[j]) ** 2 / norm(A[j]) ** 2,
        norm(A[j] @ r) ** 2 / norm(A[j]) ** 2,
        cos(C[j], r),
        norm(r @ B[j]) ** 2 / norm(B[j]) ** 2,
        norm(B[j] @ r) ** 2 / norm(B[j]) ** 2,
    ]
    print(labels[j], np.round(vals, 5))

print("\nFIXED KEY-SCORING VECTOR CROSS-HEAD COSINES")
print("Rows/columns within each layer are H0..H5")
for l in range(4):
    print("Layer", l)
    print(np.array([[cos(C[6 * l + i], C[6 * l + j]) for j in range(6)] for i in range(6)]))

print("\nLAYER 0 INPUT-WEIGHT SPECTRA")
for name in ("W_q", "W_k", "W_v", "W_fc"):
    lnn = "ln2_w.0" if name == "W_fc" else "ln1_w.0"
    m = np.asarray(W[name + ".0"], dtype=float) * np.asarray(W[lnn], dtype=float)
    u, s, vt = svd(m, full_matrices=False, check_finite=False)
    print(name, "first 48 singular values:", np.round(s[:48], 4))

print("\nL0H3 COMPOSITION INTO EACH LATER HEAD")
print(
    "Columns: head, ||QK @ OV_03|| / (||QK|| ||OV_03||), "
    "trace_product/norm_product, symmetry_cos_product, "
    "positive_real_eigenvalue_sum/absolute_eigenvalue_sum"
)
for j in range(6, 24):
    m = A[j] @ B[3]
    ev = eigvals(m, check_finite=False)
    pos = np.maximum(ev.real, 0).sum() / (np.abs(ev).sum() + 1e-30)
    print(
        labels[j],
        np.round([norm(m) / (norm(A[j]) * norm(B[3])), np.trace(m) / (norm(m) + 1e-30), cos(m, m.T), pos], 5),
    )
```