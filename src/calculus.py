import sympy as sp


def classify_critical_points(f, vars):
    grad = [sp.diff(f, v) for v in vars]
    sols = sp.solve(grad, vars, dict=True)
    hess = sp.hessian(f, vars)

    results = []
    for sol in sols:
        point = tuple(sol[v] for v in vars)
        H_val = hess.subs(sol)
        eigvals = sorted(sp.N(e) for e in H_val.eigenvals().keys())

        if all(e > 0 for e in eigvals):
            tipo = 'minimo'
        elif all(e < 0 for e in eigvals):
            tipo = 'massimo'
        else:
            tipo = 'sella'

        results.append({'point': point, 'hessian': H_val, 'eigenvalues': eigvals, 'type': tipo})
    return results
