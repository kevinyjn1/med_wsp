```python
import numpy as np
from scipy.linalg import svd, svdvals
from threadpoolctl import threadpool_limits

np.set_printoptions(precision=3, suppress=True, linewidth=190)
with threadpool_limits(limits=2):
    fc = W["W_fc.0"].astype(float)
    out = W["W_proj.0"].astype(float).T
    fn = np.linalg.norm(fc, axis=1)
    on = np.linalg.norm(out, axis=1)
    cos = np.sum(fc * out, axis=1) / (fn * on)
    ix = np.argsort(fn)[::-1]
    print("First MLP input-row norm percentiles:", np.percentile(fn, [0, 25, 50, 75, 90, 95, 98, 99, 100]))
    print("Top 60 first MLP input rows: neuron, input norm, output norm, input/output cosine")
    for i in ix[:60]:
        print(int(i), round(fn[i], 3), round(on[i], 3), round(cos[i], 3))

    print("\nMost anti-aligned first MLP neurons:")
    neg = np.argsort(cos)[:35]
    print([(int(i), round(cos[i], 3), round(fn[i], 3), round(on[i], 3)) for i in neg])
    z = fc[ix[:32]] / fn[ix[:32], None]
    gram = z @ z.T
    print(
        "Top-row pairwise cosine percentiles:",
        np.percentile(gram[np.triu_indices(32, 1)], [0, 10, 25, 50, 75, 90, 100]),
    )
    print("Singular values of normalized top32 rows:", svdvals(z))

    print("\nL0H2 positional query/key row-space relation")
    q = W["W_q.0"][128:192].astype(float) * W["ln1_w.0"]
    k = W["W_k.0"][128:192].astype(float) * W["ln1_w.0"]
    q -= q.mean(1, keepdims=True)
    k -= k.mean(1, keepdims=True)
    _, s, vt = svd(np.concatenate([q, k]), full_matrices=False)
    p = vt[:24].T
    metric = (p.T @ (q.T @ q + k.T @ k) @ p) / 2
    e, u = np.linalg.eigh(metric)
    whitening = u @ np.diag(1 / np.sqrt(e)) @ u.T
    a = whitening @ (p.T @ q.T @ k @ p) @ whitening
    u, s, vt = svd(a, full_matrices=False)
    rotation = u @ vt
    ev = np.linalg.eigvals(rotation)
    print("Whitened QK singular values:", s)
    print("Rotation eigenvalue phases:", np.sort(np.angle(ev)))
    print("Other first-layer QK products expressed against powers of this weight-derived rotation")
    powers = {r: np.linalg.matrix_power(rotation, r) for r in range(-6, 7)}
    for j in range(6):
        sl = slice(64 * j, 64 * (j + 1))
        qj = W["W_q.0"][sl].astype(float) * W["ln1_w.0"]
        kj = W["W_k.0"][sl].astype(float) * W["ln1_w.0"]
        qj -= qj.mean(1, keepdims=True)
        kj -= kj.mean(1, keepdims=True)
        aj = whitening @ (p.T @ qj.T @ kj @ p) @ whitening
        scores = [
            np.sum(aj * powers[r]) / (np.linalg.norm(aj) * np.linalg.norm(powers[r])) for r in range(-6, 7)
        ]
        print(f"L0H{j} cosines for powers -6..6:", np.round(scores, 3))
```
