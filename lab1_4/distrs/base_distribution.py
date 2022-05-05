from abc import abstractmethod
import numpy as np

class BaseDistribution:

    def __init__(self):
        self._rng = np.random.default_rng()

    @abstractmethod
    def get_rvs(self, num):
        pass

    @abstractmethod
    def get_pdf(self, x):
        pass

    @abstractmethod
    def get_cdf(self, x):
        pass
