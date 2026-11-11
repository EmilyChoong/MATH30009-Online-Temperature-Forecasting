import numpy as np


class ARCOR:
    """
    ARCOR (Adaptive Regularization with Covariance Reset), Moroshko et al. (2015), Table 2.
    AROWR plus two changes for non-stationary data:
      - covariance reset: if the smallest eigenvalue of the candidate Sigma drops below
        Lambda_i = 1 / (i^(q-1) + 1), set Sigma = I and move to the next segment i + 1;
      - projection: keep the weights inside the ball ||w|| <= R_B.
    Ported from Aaron's arcor() (arcor_algorithm_v2.Rmd).
    """

    def __init__(self, p, r=1.0, q=2.0, R_B=100.0):
        self.r, self.q, self.R_B = r, q, R_B
        self.w = np.zeros(p)              # w_0 = 0
        self.Sigma = np.eye(p)            # Sigma_0 = I
        self.i = 1                        # segment index
        self.t = 0                        # number of updates so far (= time index in run_online)
        self.reset_times = []             # update indices t at which Sigma was reset
        self.projection_times = []        # update indices t at which w was projected

    @property
    def n_resets(self):
        return len(self.reset_times)

    @property
    def n_projections(self):
        return len(self.projection_times)

    def predict(self, x):
        return x @ self.w                 # y_hat_t = x_t * w_{t-1}

    def update(self, x, y):
        Sx = self.Sigma @ x               # Sigma_{t-1} x_t
        denom = self.r + x @ Sx           # r + x_t * Sigma_{t-1} * x_t

        # candidate covariance and weights (AROWR step)
        Sigma_tilde = self.Sigma - np.outer(Sx, Sx) / denom
        Sigma_tilde = (Sigma_tilde + Sigma_tilde.T) / 2            # keep it exactly symmetric
        w_tilde = self.w + (y - x @ self.w) / denom * Sx

        # covariance reset
        Lambda_i = 1 / (self.i ** (self.q - 1) + 1)
        if np.linalg.eigvalsh(Sigma_tilde)[0] < Lambda_i:
            self.Sigma = np.eye(len(x))
            self.i += 1
            self.reset_times.append(self.t)
        else:
            self.Sigma = Sigma_tilde

        # projection onto ||w|| <= R_B in the Sigma^{-1} norm
        if np.linalg.norm(w_tilde) <= self.R_B:
            self.w = w_tilde
        else:
            self.projection_times.append(self.t)
            self.w = self._project(w_tilde)

        self.t += 1

    def _project(self, w_tilde):
        """w = (I + alpha Sigma)^{-1} w_tilde with alpha found by bisection so that ||w|| = R_B."""
        eigenvalues, Q = np.linalg.eigh(self.Sigma)
        z = Q.T @ w_tilde
        norm = lambda a: np.linalg.norm(z / (1 + a * eigenvalues))

        low, high = 0.0, 1.0
        while norm(high) > self.R_B:      # find an upper bound for alpha
            high *= 2
        for _ in range(60):               # bisection
            alpha = (low + high) / 2
            if norm(alpha) > self.R_B:
                low = alpha
            else:
                high = alpha
        return Q @ (z / (1 + high * eigenvalues))


# parameter grid searched by online.tune (486 combinations).
# Extended after checking grid edges: for short k the tuning MSE is flat for r <= 1e-4
# and for every q (ARCOR resets every hour); for long k it is flat for q <= 1.01.
# R_B = 25 is an interior optimum (15, 20 and 30 are all worse).
GRID = {
    "r":   [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2],
    "q":   [1.001, 1.01, 1.05, 1.1, 1.25, 1.5, 2, 2.5, 3],
    "R_B": [15, 20, 25, 30, 50, 100],
}
