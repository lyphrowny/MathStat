from .base_distribution import *


class Normal(BaseDistribution):

    def __init__(self, mu, sigma):
        super().__init__()
        self._mu = mu
        self._sigma = sigma

    def get_rvs(self, num):
        return self._rng.normal(self._mu, self._sigma, num)

    def get_pdf(self, x):
        return 1 / (self._sigma * np.sqrt(2 * np.pi)) * \
               np.exp(-np.power(x - self._mu, 2) / (2 * np.power(self._sigma, 2)))

    def get_cdf(self, x):
        return 0.5 * (1 + np.erf((x - self._mu) / (self._sigma * np.sqrt(2))))
