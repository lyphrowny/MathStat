from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from lab2 import _z_p
from setup import set_up


def _q1(xs):
    return _z_p(xs, 0.25)


def _q3(xs):
    return _z_p(xs, 0.75)


def _boxplot(distr, ps_num, plot_dir):
    title = distr.__class__.__name__
    plt.title(title)
    data = list(map(distr.get_rvs, ps_num))
    plt.boxplot(data, labels=ps_num, vert=False)
    plt.xlabel("x")
    plt.ylabel("n")

    plt.savefig(plot_dir.joinpath(title))
    plt.close()


def _outlier(distr, ps_num, times=1000):
    r = []
    for p_num in ps_num:
        cnt = 0
        for _ in range(times):
            x = np.sort(distr.get_rvs(p_num))
            q1, q3 = _q1(x), _q3(x)
            lo, hi = q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)
            cnt += len(x[np.logical_or(x < lo, hi < x)])
        r.append(f"{distr.__class__.__name__}, n={p_num} & {cnt / (times * p_num)} \\\\\n")
    return r


def lab3(distrs, ps_num, plot_dir, table_dir, times=1000):
    if not plot_dir.exists():
        plot_dir.mkdir()
    if not table_dir.exists():
        table_dir.mkdir(parents=True)
    table = []
    for distr in distrs:
        _boxplot(distr, ps_num, plot_dir)
        table.extend(_outlier(distr, ps_num, times))

    dest = table_dir.joinpath("outliers")
    dest.write_text("".join(table))


if __name__ == "__main__":
    distrs = set_up()
    lab3(distrs, [20, 100], Path(""), Path(""))
