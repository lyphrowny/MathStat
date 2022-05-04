import numpy as np

from distrs import *


def set_up():
    return [
        Normal(0, 1),
        Cauchy(0, 1),
        Laplace(0, 1 / np.sqrt(2)),
        Poisson(10),
        Uniform(-np.sqrt(3), np.sqrt(3)),
    ]
