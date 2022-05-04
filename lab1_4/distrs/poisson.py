import scipy.special as sps

from .base_distribution import *


class Poisson(BaseDistribution):

    def __init__(self, _lambda):
        super().__init__()
        self._lambda = _lambda

    def get_rvs(self, num):
        return self._rng.poisson(self._lambda, num)

    def get_pdf(self, k):
        return np.float_power(self._lambda, k) / sps.factorial(k) * np.exp(-self._lambda)
