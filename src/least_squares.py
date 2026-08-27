import numpy as np
import scipy.linalg as la


def qr_lstsq(A, b):
    """Soluzione ai minimi quadrati di A x = b tramite fattorizzazione QR economica (A = QR, poi R x = Q^T b)."""
    Q, R = la.qr(A, mode="economic")
    return la.solve(R, Q.T @ b)


def qr_polyfit(x, y, degree):
    """Fit polinomiale di grado `degree` sui punti (x, y), risolto via QR sulla matrice di Vandermonde di x."""
    A = np.vander(x, degree + 1)
    return qr_lstsq(A, y)


def givens_qr_update(R, z, a_new, b_new):
    """Aggiorna una fattorizzazione QR economica (R, z = Q^T b) incorporando una nuova riga (a_new, b_new) tramite una sequenza di rotazioni di Givens."""
    R = R.copy()
    z = z.copy()
    a = a_new.copy()
    b = b_new

    for j in range(R.shape[0]):
        row_j = R[j, :].copy()
        x, yj = row_j[j], a[j]
        r = np.hypot(x, yj)
        c, s = x / r, yj / r

        R[j, :] = c * row_j + s * a
        a = -s * row_j + c * a

        z_j = z[j]
        z[j] = c * z_j + s * b
        b = -s * z_j + c * b

    return R, z


def vandermonde_conditioning(x, degrees):
    """Numero di condizionamento (norma 2) della matrice di Vandermonde di x, per ciascun grado in `degrees`."""
    return [np.linalg.cond(np.vander(x, d + 1)) for d in degrees]


def pcr_solution_path(A, b):
    """Serie di soluzioni troncate x_k (k=1..numero di colonne di A) del sistema A x = b via SVD, con il residuo relativo ||A x_k - b||/||b|| a ciascun k."""
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    n_cols = A.shape[1]
    x_k = np.zeros(n_cols)
    x_k_list, residuals = [], []
    for i in range(n_cols):
        x_k = x_k + (U[:, i] @ b / s[i]) * Vt[i, :]
        residuals.append(np.linalg.norm(A @ x_k - b) / np.linalg.norm(b))
        x_k_list.append(x_k.copy())
    return x_k_list, residuals
