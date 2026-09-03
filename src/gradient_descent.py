import numpy as np


def gradient_descent(grad, x0, lr, atol=1e-8, rtol=1e-8, nmax=1000):
    x0 = np.array(x0, dtype=float)
    all_x = [x0.copy()]
    g0 = np.array(grad(x0)).flatten()
    x1 = x0 - lr*g0
    err = np.linalg.norm(x1 - x0) / (1 + rtol/atol*np.linalg.norm(x0))
    n_it = 1
    while err > atol and n_it < nmax:
        x0 = x1
        all_x.append(x0.copy())
        g0 = np.array(grad(x0)).flatten()
        x1 = x0 - lr*g0
        err = np.linalg.norm(x1 - x0) / (1 + rtol/atol*np.linalg.norm(x0))
        n_it += 1
    all_x.append(x1.copy())
    return x1, np.array(grad(x1)).flatten(), n_it, np.array(all_x)
