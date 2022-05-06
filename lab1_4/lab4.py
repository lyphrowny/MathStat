from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

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
    if not emp_dir.exists():
        emp_dir.mkdir(parents=True)
    fig.savefig(emp_dir.joinpath(title))
    plt.close(fig)


def _nuclear(distr, ps_num, nuc_dir: Path):
    pass


def lab4(distrs, ps_num, emp_dir: Path, nuc_dir: Path):
    for distr in distrs:
        # _empirical(distr, ps_num, emp_dir)
        _nuclear(distr, ps_num, nuc_dir)


if __name__ == "__main__":
    distrs = set_up()
    lab4(distrs, [20, 60, 100])
