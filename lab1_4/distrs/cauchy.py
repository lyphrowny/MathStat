from .base_distribution import *


class Cauchy(BaseDistribution):

    def __init__(self, x0, gamma):
        super().__init__()
        self._x0 = x0
        self._gamma = gamma

    def get_rvs(self, num):
        return self._rng.standard_cauchy(num)

    def get_pdf(self, x):
        return 1 / (np.pi * self._gamma * (1 + np.power((x - self._x0) / self._gamma, 2)))

    def get_cdf(self, x):
        return 1 / np.pi * np.atan((x - self._x0) / self._gamma) + 0.5
