import numpy as np


def svd_truncate(A, k):
    """Approssimazione di rango k di A tramite SVD troncata: A_k = U[:,:k] @ diag(s[:k]) @ Vt[:k,:]."""
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    return U[:, :k] @ np.diag(s[:k]) @ Vt[:k, :]


def reconstruction_error(A, A_k, mask=None):
    """Norma dell'errore di ricostruzione A_k - A, opzionalmente solo sulle voci indicate da mask."""
    diff = A_k - A
    if mask is not None:
        diff = diff[mask]
    return np.linalg.norm(diff)
