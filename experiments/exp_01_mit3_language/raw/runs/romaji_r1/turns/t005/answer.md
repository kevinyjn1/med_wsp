```python
import numpy as np
from scipy.linalg import svd, eig

np.set_printoptions(precision=4, suppress=True, linewidth=170)
d = 384
P = np.eye(d) - np.ones((d, d)) / d
g = W["ln1_w.0"].astype(float)
Q = [(W["W_q.0"][64 * h : 64 * h + 64].astype(float) * g) @ P for h in range(6)]
K = [(W["W_k.0"][64 * h : 64 * h + 64].astype(float) * g) @ P for h in range(6)]
V = [(W["W_v.0"][64 * h : 64 * h + 64].astype(float) * g) @ P for h in range(6)]
A = [q.T @ k / 8 for q, k in zip(Q, K)]
_, sv, vt = svd(np.concatenate([V[0], V[3]]), full_matrices=False)
print("V of content-carrying H0/H3 singular values", sv[:50])
for r in [16, 24, 28, 32]:
    C = vt[:r].T
    R = P - C @ C.T
    q = Q[3] @ R
    k = K[3] @ R
    u, s, v = svd(np.concatenate([q, k]), full_matrices=False)
    print("\nRemoved value-input rank", r, "remaining QK3 spectrum", s[:40])
    T = v[:32].T / s[:32]
    Z = T.T @ A[3] @ T
    Z0 = T.T @ A[0] @ T
    e, E = eig(Z)
    ix = np.argsort(np.angle(e))
    e = e[ix]
    E = E[:, ix]
    transformed = np.linalg.pinv(E) @ Z0 @ E
    diag = np.diag(transformed)
    print("H3/H0 eigenbasis diagonal (angle3, magnitude3, angle0, magnitude0):")
    for j in np.where(np.imag(e) > 1e-5)[0]:
        print(np.round([np.angle(e[j]), abs(e[j]), np.angle(diag[j]), abs(diag[j])], 4))
    print("H0 diagonal fraction", np.linalg.norm(np.diag(diag)) ** 2 / np.linalg.norm(transformed) ** 2)
    print(
        "H0 whitened commutator ratio",
        np.linalg.norm(Z @ Z0 - Z0 @ Z) / (np.linalg.norm(Z) * np.linalg.norm(Z0)),
    )
    z = np.eye(len(Z))
    for power in range(1, 7):
        z = z @ Z
        print("power", power, "cos with H0", np.sum(z * Z0) / np.linalg.norm(z) / np.linalg.norm(Z0))

print("\nL0 QK versus V subspace intersections")
for h in [0, 3]:
    uq, sq, vq = svd(np.concatenate([Q[h], K[h]]), full_matrices=False)
    for r in [16, 24, 32, 48]:
        print("head", h, "r", r, "overlap singular values", svd(vq[:r] @ vt[:24].T, compute_uv=False)[:12])
```
