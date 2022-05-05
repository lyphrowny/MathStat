import numpy as np

from distrs import *


def set_up():
    return [
        Normal(0, 1, [-4, 4]),
        Cauchy(0, 1, [-4, 4]),
        Laplace(0, 1 / np.sqrt(2), [-4, 4]),
        Poisson(10, [6, 14]),
        Uniform(-np.sqrt(3), np.sqrt(3), [-4, 4]),
    ]
