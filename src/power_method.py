import numpy as np


def power_method(u0, A, tol, it_max):
    n_it = 0
    u1 = A @ u0
    lam1 = u0 @ u1 / (u0 @ u0)

    approx = [lam1]
    err = [1.0]

    while err[-1] > tol and n_it < it_max:
        u0 = u1
        lam0 = lam1
        u1 = A @ u0
        lam1 = u0 @ u1 / (u0 @ u0)
        approx.append(lam1)
        err.append(abs(lam1 - lam0))
        n_it += 1

    return lam1, u1, n_it, err, approx


def power_method_norm(u0, A, tol, it_max):
    n_it = 1
    z0 = u0 / np.linalg.norm(u0)
    u1 = A @ z0
    z1 = u1 / np.linalg.norm(u1)
    lam1 = z0 @ u1 / (z0 @ z0)

    approx = [lam1]
    err = [1.0]

    while err[-1] > tol and n_it < it_max:
        lam0 = lam1
        z0 = z1
        u1 = A @ z0
        z1 = u1 / np.linalg.norm(u1)
        lam1 = z0 @ u1 / (z0 @ z0)
        approx.append(lam1)
        err.append(abs(lam1 - lam0))
        n_it += 1

    return lam1, z1, n_it - 1, err, approx
