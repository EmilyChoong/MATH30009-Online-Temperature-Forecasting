import numpy as np


class RLS:
    """
    RLS (Recursive Least Squares with forgetting factor r), Moroshko et al. (2015), Table 1.
    Minimises sum_s r^{t-s} (y_s - x_s^T w)^2, so r < 1 down-weights old data.
      Initialise:  w_0 = 0, Sigma_0 = I
      Update:      w_t = w_{t-1} + (y_t - x_t^T w_{t-1}) Sigma_{t-1} x_t / (r + x_t^T Sigma_{t-1} x_t)
                   Sigma_t = (Sigma_{t-1} - Sigma_{t-1} x_t x_t^T Sigma_{t-1} / (r + x_t^T Sigma_{t-1} x_t)) / r
    Ported from 'RLS and CR-RLS.py' (group member's script).
    """

    def __init__(self, p, r=0.99):
        self.r = r
        self.w = np.zeros(p)              # w_0 = 0
        self.Sigma = np.eye(p)            # Sigma_0 = I
        self.t = 0                        # number of updates so far

    def predict(self, x):
        return x @ self.w

    def update(self, x, y):
        Sx = self.Sigma @ x               # Sigma_{t-1} x_t
        denom = self.r + x @ Sx           # r + x_t^T Sigma_{t-1} x_t

        self.w = self.w + (y - x @ self.w) / denom * Sx
        self.Sigma = (self.Sigma - np.outer(Sx, Sx) / denom) / self.r
        self.Sigma = (self.Sigma + self.Sigma.T) / 2
        self.t += 1
        self._after_update()

    def _after_update(self):
        pass                              # hook used by CR-RLS


class CRRLS(RLS):
    """
    CR-RLS (Covariance Reset RLS), Moroshko et al. (2015), Section 2:
    RLS, but Sigma is reset to I every T0 rounds.
    """

    def __init__(self, p, r=0.99, T0=168):
        super().__init__(p, r)
        self.T0 = T0
        self.reset_times = []             # update indices t at which Sigma was reset

    def _after_update(self):
        if self.t % self.T0 == 0:
            self.Sigma = np.eye(len(self.w))
            self.reset_times.append(self.t - 1)


# parameter grids searched by online.tune
MEMORY = [2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000]            # hours
GRID = {"r": [1 - 1/m for m in MEMORY] + [1.0]}
CRRLS_GRID = {
    "r":  GRID["r"],
    "T0": [24, 48, 72, 168, 336, 720, 1440, 2880],   # finite only: T0 = inf is plain RLS
}
