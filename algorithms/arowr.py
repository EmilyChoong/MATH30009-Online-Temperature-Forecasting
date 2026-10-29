import numpy as np

class AROWR:
    """
    AROWR (Vaits & Crammer, 2011), as given in Moroshko et al. (2015), Table 1.
    Online linear regression y_t ≈ x_tᵀw, where x_t is the natural cubic spline basis at hour h_t.
    """

    def __init__(self, p, r=1.0):
        self.r = r
        self.w = np.zeros(p)              # spline weights, w_0 = 0
        self.Sigma = np.eye(p)            # Sigma_0 = I

    def predict(self, x):
        return x @ self.w                 # y_hat_t = x_t * w_{t-1}

    def update(self, x, y):
        Sx =  self.Sigma @ x              # Sigma_{t-1} x_t
        denom = self.r + x @ Sx           # r + x_t * Sigma_{t-1} * x_t

        # Update the weights and covariance matrix using the AROWR update rules
        self.w = self.w + (y - x @ self.w) / denom * Sx        # eq. (5)
        self.Sigma = self.Sigma - np.outer(Sx, Sx) / denom     # eq. (6)



# parameter grid searched by online.tune (log-spaced)
GRID = {"r": [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10]}
