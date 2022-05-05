from .base_distribution import *


class Uniform(BaseDistribution):

    def __init__(self, a, b):
        super().__init__()
        self._a = a
        self._b = b

    def get_rvs(self, num):
        return self._rng.uniform(self._a, self._b, num)

    def get_pdf(self, x):
        return [0, 1 / (self._b - self._a)][self._a <= x <= self._b]

    def get_cdf(self, x):
        return 0 if x < self._a else (1 if x > self._b else (x - self._a) / (self._b - self._a))
