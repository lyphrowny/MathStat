from abc import abstractmethod
import numpy as np


class BaseDistribution:

    def __init__(self, lims):
        self._rng = np.random.default_rng()
        self._lo, self._hi = sorted(lims)

    def get_lims(self):
        return self._lo, self._hi

    def trunc_to_lims(self, xs):
        return xs[np.logical_and(self._lo <= xs, xs <= self._hi)]

    @abstractmethod
    def get_rvs(self, num):
        pass

    @abstractmethod
    def get_pdf(self, x):
        pass

    @abstractmethod
    def get_cdf(self, x):
        pass
