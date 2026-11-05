import numpy as np


class AAR:
    """
    AAR (Aggregating Algorithm for Regression, Vovk 2001), as given in Moroshko et al. (2015), Table 1.
    Stationary ridge-type regression: LASER with c = infinity.
      Initialise:  w_0 = 0, Sigma_0 = b^{-1} I
      Update:      Sigma_t^{-1} = Sigma_{t-1}^{-1} + x_t x_t^T
                   w_t = w_{t-1} + (y_t - x_t^T w_{t-1}) Sigma_{t-1} x_t / (1 + x_t^T Sigma_{t-1} x_t)
    Ported from aar.py (group member's script).
    """

    def __init__(self, p, b=0.01):
        self.b = b
        self.w = np.zeros(p)              # w_0 = 0
        self.Sigma = np.eye(p) / b        # Sigma_0 = b^{-1} I

    def predict(self, x):
        return x @ self.w                 # Task 3 forecast x_{t+k}^T w_t (not the shrunk Table 1 prediction)

    def update(self, x, y):
        Sx = self.Sigma @ x               # Sigma_{t-1} x_t
        denom = 1 + x @ Sx                # 1 + x_t^T Sigma_{t-1} x_t

        self.w = self.w + (y - x @ self.w) / denom * Sx
        self.Sigma = self.Sigma - np.outer(Sx, Sx) / denom      # Sherman-Morrison form of Sigma_t^{-1} = Sigma_{t-1}^{-1} + x x^T
        self.Sigma = (self.Sigma + self.Sigma.T) / 2            # keep it exactly symmetric


# parameter grid searched by online.tune
GRID = {"b": [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2]}
