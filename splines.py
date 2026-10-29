import numpy as np

def ncs_basis(h, p, scale=True):
    """Natural cubic spline basis at hours h, with p knots evenly spaces over hours 0-23.
    This function generates the basis functions & 
    returns a (24 (hours of the day), p) matrix whose row t = x_t (input vector)."""

    h = np.asarray(h, dtype=float)
    knots = np.linspace(0, 23, p)
    if scale:                                               # rescale hours to [0, 1] -> 0/23, ... 23/23
        h, knots = h/23, knots/23                           # rescale hours to [0,1] so all basis functions are on a similar scale

    pos3 = lambda z: np.maximum(z, 0) ** 3                  # ensures the basis function is zero until it's corresponding knot & smoothing
    d = lambda k: ( pos3(h - knots[k]) - pos3(h - knots[-1])) / (knots[-1] - knots[k])
                                                            # d: our building blocks for each basis function
    cols = [np.ones_like(h), h] + [d(k) - d(p - 2) for k in range(p - 2)]
    return np.column_stack(cols)

if __name__ == "__main__":
    X = ncs_basis(np.arange(24), p=5)
    print(X.shape)                                          # shape (24, 5): one row per hour
