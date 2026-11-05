import numpy as np


class LASER:
    """
    LASER (Last Step Adaptive Regressor), Moroshko et al. (2015), Table 2.
    AAR with an extra step that adds c^{-1} I to Sigma every round, so old data are
    slowly forgotten (c = infinity gives AAR). Requires 0 < b < c.
      Initialise:  w_0 = 0, Sigma_0 = (c - b) / (b c) I
      Update:      A = Sigma_{t-1} + c^{-1} I
                   Sigma_t^{-1} = A^{-1} + x_t x_t^T
                   w_t = w_{t-1} + (y_t - x_t^T w_{t-1}) A x_t / (1 + x_t^T A x_t)
    Ported from laser.py (group member's script).
    """

    def __init__(self, p, b=0.01, c=100.0):
        if not 0 < b < c:
            raise ValueError("LASER requires 0 < b < c")
        self.b, self.c = b, c
        self.w = np.zeros(p)                              # w_0 = 0
        self.Sigma = (c - b) / (b * c) * np.eye(p)        # Sigma_0
        self.I_over_c = np.eye(p) / c

    def predict(self, x):
        return x @ self.w                 # Task 3 forecast x_{t+k}^T w_t (not the min-max Table 2 prediction)

    def update(self, x, y):
        A = self.Sigma + self.I_over_c    # Sigma_{t-1} + c^{-1} I
        Ax = A @ x
        denom = 1 + x @ Ax                # 1 + x_t^T A x_t

        self.w = self.w + (y - x @ self.w) / denom * Ax
        self.Sigma = A - np.outer(Ax, Ax) / denom        # Sherman-Morrison form of Sigma_t^{-1} = A^{-1} + x x^T
        self.Sigma = (self.Sigma + self.Sigma.T) / 2


# parameter grid searched by online.tune (combinations with b >= c are skipped).
# Widened from the original script, whose best p = 5, 6 values sat on the grid's lower edge.
GRID = {
    "b": [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2],
    "c": [1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4],
}
