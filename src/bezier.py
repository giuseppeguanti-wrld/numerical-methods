import numpy as np


def bezier_cubic(P0, P1, P2, P3, n=100):
    """Valuta la Bezier cubica con punti di controllo P0..P3 su n punti equispaziati in t."""
    P0, P1, P2, P3 = (np.asarray(P, dtype=float) for P in (P0, P1, P2, P3))
    t = np.linspace(0, 1, n)
    B = (
        (1 - t) ** 3 * P0[:, None]
        + 3 * (1 - t) ** 2 * t * P1[:, None]
        + 3 * (1 - t) * t ** 2 * P2[:, None]
        + t ** 3 * P3[:, None]
    )
    return B.T, t


def bezier_cubic_d1(P0, P1, P2, P3, t):
    """Derivata prima della Bezier cubica, formula chiusa."""
    P0, P1, P2, P3 = (np.asarray(P, dtype=float) for P in (P0, P1, P2, P3))
    d1 = (
        3 * (1 - t) ** 2 * (P1 - P0)[:, None]
        + 6 * (1 - t) * t * (P2 - P1)[:, None]
        + 3 * t ** 2 * (P3 - P2)[:, None]
    )
    return d1.T


def bezier_cubic_d2(P0, P1, P2, P3, t):
    """Derivata seconda della Bezier cubica, formula chiusa."""
    P0, P1, P2, P3 = (np.asarray(P, dtype=float) for P in (P0, P1, P2, P3))
    d2 = (
        6 * (1 - t) * (P2 - 2 * P1 + P0)[:, None]
        + 6 * t * (P3 - 2 * P2 + P1)[:, None]
    )
    return d2.T


def curvature(P0, P1, P2, P3, t):
    """Curvatura k(t) = |x'y'' - y'x''| / (x'^2+y'^2)^{3/2}."""
    d1 = bezier_cubic_d1(P0, P1, P2, P3, t)
    d2 = bezier_cubic_d2(P0, P1, P2, P3, t)
    num = np.abs(d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0])
    den = (d1[:, 0] ** 2 + d1[:, 1] ** 2) ** 1.5
    return num / den
