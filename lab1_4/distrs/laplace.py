from .base_distribution import *


class Laplace(BaseDistribution):

    def __init__(self, beta, alpha):
        super().__init__()
        self._beta = beta
        # it's 1 / alpha actually
        self._alpha = 1 / alpha

    def get_rvs(self, num):
        return self._rng.laplace(self._beta, self._alpha, num)

    def get_pdf(self, x):
        return np.exp(-np.fabs(x - self._beta) / self._alpha) / (2 * self._alpha)
