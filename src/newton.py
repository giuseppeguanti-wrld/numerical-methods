import numpy as np
import numpy.linalg as nl


def newton(f, df, x0, atol=1e-8, rtol=1e-8, nmax=100):
    x1 = x0 - f(x0) / df(x0)
    all_x = np.array([x0])
    n_it = 1
    while abs(x1 - x0) / (1 + rtol / atol * abs(x0)) > atol and n_it < nmax:
        x0 = x1
        x1 = x0 - f(x0) / df(x0)
        all_x = np.hstack((all_x, x1))
        n_it += 1
    return x0, f(x0), n_it, all_x


def newton_vett(f, df, x0, atol=1e-8, rtol=1e-8, nmax=100):
    f0 = f(x0)
    df0 = df(x0)
    dx = nl.solve(df0, f0)
    x1 = x0 - dx
    err = nl.norm(dx) / (1 + rtol / atol * nl.norm(x0))
    all_err = np.array([err])
    n_it = 1
    while err > atol and n_it < nmax:
        x0 = x1
        f0 = f(x0)
        df0 = df(x0)
        dx = nl.solve(df0, f0)
        x1 = x0 - dx
        err = nl.norm(dx) / (1 + rtol / atol * nl.norm(x0))
        all_err = np.hstack((all_err, err))
        n_it += 1
    return x0, f0, df0, n_it, all_err
