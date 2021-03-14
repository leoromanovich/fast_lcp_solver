import numpy as np

__all__ = ['solveLCP']


def solveLCP(M: np.ndarray, q: np.ndarray) -> np.ndarray:
    n = q.shape[0]
    p = np.zeros(n)
    I = np.identity(n)

    z = np.sign(np.linalg.inv(I + M) @ q)
    Tz = np.diag(z.reshape(-1))
    H = np.linalg.inv((I + M) + (I - M) @ Tz)
    x = H @ q
    C = -H @ (I - M)

    xz = x * z
    neg_xz_vals = np.where(xz < 0)[0]
    while neg_xz_vals.size > 0:
        j = neg_xz_vals[0]

        if 1 + (2 * z[j] * C[j, j]) <= 0:
            return []

        p[j] = p[j] + 1

        if np.log2(p[j]) > n - j:
            return []

        if 1 + 2 * z[j] * C[j, j] > 0:
            ej = I[:, j].reshape(n, 1)
            Tz = Tz - 2 * z[j] * ej * ej.T
            z[j] = -z[j]
            alpha = (2 * z[j])/(1 - 2 * z[j] * C[j, j])
            x = x + alpha * x[j] * C[:, j].reshape(-1, 1)
            C = C + alpha * C[:, j] @ C[j, :]

        xz = x * z
        neg_xz_vals = np.where(xz < 0)[0]
    return np.abs(x) - x
