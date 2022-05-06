from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

from lab1_4.lab2 import _mean
from setup import set_up


def _get_freqs(rvs, points):
    # len of `rvs` might differ from len of points
    # as the `rvs` are truncated to their limits
    n = len(points)
    freq = [0] * n
    idx = 0
    for num, point in enumerate(rvs):
        # cumulative number of points
        while idx < n and point >= points[idx]:
            freq[idx] = num / n
            idx += 1
    # fill the rest
    for i in range(idx, n):
        freq[i] = 1
    return freq


def _empirical(distr, ps_num, emp_dir: Path):
    fig, axs = plt.subplots(1, len(ps_num), figsize=(10.5, 4), tight_layout=True)
    title = distr.__class__.__name__
    fig.suptitle(title)
    for p_num, ax in zip(ps_num, axs):
        x = np.sort(distr.trunc_to_lims(distr.get_rvs(p_num)))
        points = np.linspace(*distr.get_lims(), num=p_num)
        ax.plot(points, [distr.get_cdf(p) for p in points])
        ax.step(points, _get_freqs(x, points))

        ax.set_title(f"n={p_num}")
        ax.legend("cdf empirical".split())
        ax.set_xlabel("x")
        ax.set_ylabel("F(x)")

    fig.savefig(emp_dir.joinpath(title))
    plt.close(fig)


def _h(x):
    std = lambda xs: np.sqrt(_mean(np.power(xs, 2)) - np.power(_mean(xs), 2))
    return 1.06 * std(x) * np.float_power(len(x), -0.2)


def _f(p, xs, h):
    K = lambda u: np.exp(-np.power(u, 2) / 2) / np.sqrt(2 * np.pi)
    n = len(xs)
    return sum(K((p - xs[i]) / h) for i in range(1, n)) / (n * h)


def _nuclear(distr, ps_num, nuc_dir: Path):
    for p_num in ps_num:
        fig, axs = plt.subplots(1, 3, figsize=(10.5, 4), tight_layout=True)
        title = f"{distr.__class__.__name__} n={p_num}"
        fig.suptitle(title)

        x = np.sort(distr.trunc_to_lims(distr.get_rvs(p_num)))
        points = np.linspace(*distr.get_lims(), num=p_num)
        hn = _h(x)
        hs = dict(zip("0.5hₙ hₙ 2hₙ".split(), [hn / 2, hn, 2 * hn]))
        for (h_name, h), ax in zip(hs.items(), axs):
            ax.plot(points, [distr.get_pdf(p) for p in points])
            ax.plot(points, [_f(p, x, h) for p in points])

            ax.set_title(h_name)
            ax.legend("pdf nuc".split())
            ax.set_xlabel("x")
            ax.set_ylabel("f(x)")
        fig.savefig(nuc_dir.joinpath(title))
        plt.close(fig)


def lab4(distrs, ps_num, emp_dir: Path, nuc_dir: Path):
    if not emp_dir.exists():
        emp_dir.mkdir(parents=True)
    if not nuc_dir.exists():
        nuc_dir.mkdir(parents=True)

    for distr in distrs:
        _empirical(distr, ps_num, emp_dir)
        _nuclear(distr, ps_num, nuc_dir)


if __name__ == "__main__":
    distrs = set_up()
    lab4(distrs, [20, 60, 100], Path(""), Path(""))
