import sympy as sp


def to_numeric(expr, vars):
    return sp.lambdify(vars, expr, 'numpy')
