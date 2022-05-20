from matplotlib import pyplot as plt
from numpy.random import standard_normal
from pathlib import Path
from scipy.optimize import minimize
import numpy as np


def _deviance(y_orig, x, b0, b1):
    return sum((_y - _x * b1 - b0) ** 2 for _y, _x in zip(y_orig, x))


def _least_sq(x, y):
    x_mean, y_mean = np.mean(x), np.mean(y)
    b1 = (np.mean(x * y) - x_mean * y_mean) / (np.mean(x * x) - x_mean ** 2)
    b0 = y_mean - b1 * x_mean
    return b0, b1


def _least_mod(x, y):
    bs = minimize(lambda bs, x, y: sum(abs(y - bs[0] - bs[1] * x)), np.array([0, 1]), args=(x, y), method="COBYLA").x
    return bs


def _plot(x, y, title, fig_path: Path, tol=2):
    fig, ax = plt.subplots()
    ax.set(xlabel="x", ylabel="y", title=title)
    ax.scatter(x, y, label="data")
    ax.plot(x, [*map(_func, x)], label="orig", color="purple")
    print(title)
    for meth in (_least_sq, _least_mod):
        b0, b1 = meth(x, y)
        meth_name = meth.__name__[1:]
        print(f"{meth_name} deviance: {_deviance(y, x, b0, b1):.{tol}f}")
        print(f"{meth_name} b0: {b0:.{tol}}, b1: {b1:.{tol}f}\n")
        ax.plot(x, b1 * x + np.full(len(x), b0), label=meth_name)
    ax.legend()
    fig.show()

    if not fig_path.exists():
        fig_path.mkdir(parents=True)
    fig.savefig(fig_path.joinpath(title))


def _func(x):
    return 2 + 2 * x


def lab6(data_range, fig_path: Path, func=None):
    if func:
        # change the func above to the passed one
        globals()["_func"] = func
    x = np.arange(*data_range)
    y = np.array([_func(_x) + _e for _x, _e in zip(x, standard_normal(len(x)))])
    y_d = np.array([_func(_x) + _e for _x, _e in zip(x, standard_normal(len(x)))])
    y_d[0], y_d[~0] = 10, -10

    _plot(x, y, "no disturbance", fig_path)
    _plot(x, y_d, "with disturbance", fig_path)


if __name__ == "__main__":
    lab6((-1.8, 2, 0.2), Path("figs/lab6"))
