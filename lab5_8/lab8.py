from collections import defaultdict
from pathlib import Path

from matplotlib import pyplot as plt
from scipy.stats import chi2, t
import numpy as np

from lab5_8.table_utils import caplab, SubTable, Table


def norm_m(n, a):
    distr = np.random.standard_normal(n)
    mean = np.mean(distr)
    std = np.std(distr)
    coeff = (std * t.ppf(1 - a / 2, n - 1)) / np.sqrt(n - 1)
    return [mean - coeff, mean + coeff]


def norm_s(n, a):
    distr = np.random.standard_normal(n)
    std = np.std(distr)
    return [std * np.sqrt(n) / np.sqrt(chi2.ppf(1 - a / 2, n - 1)), std * np.sqrt(n) / np.sqrt(chi2.ppf(a / 2, n - 1))]


def as_m(n, a):
    distr = np.random.standard_normal(n)
    mean = np.mean(distr)
    std = np.std(distr)
    u = np.quantile(distr, 1 - a / 2)
    coeff = std * u / np.sqrt(n)
    return [mean - coeff, mean + coeff]


def as_s(n, a):
    distr = np.random.standard_normal(n)
    mean = np.mean(distr)
    std = np.std(distr)
    u = np.quantile(distr, 1 - a / 2)
    m = sum((_x - mean) ** 4 for _x in distr) / n
    e = m / std ** 4 - 3
    U = u * np.sqrt((e + 2) / n)
    return [std * np.power(1 + U, -0.5), std * np.power(1 - U, -0.5)]


def _gen_table(nums, a, meth_name, cplb, tab_path: Path, tol=2):
    _trunc = lambda v, tol=tol: f"{v:.{tol}f}"
    _prettify = lambda lims, name: (lambda *l: f"{l[0]} < ${name}$ < {l[1]}")(*map(_trunc, lims))
    letters = dict(zip("ms", "mu sigma".split()))
    n_cols = len(letters) + 1

    subtbls = []
    for n in nums:
        hdrs = [f"n={n}"]
        lines = [""]
        for let in letters:
            p_let = f"\\{letters[let]}"
            hdrs.append(p_let)
            lines.append(_prettify(eval(f"{meth_name}_{let}({n}, {a})"), p_let))
        subtbls.append(SubTable(hdrs, lines))
    tbl = Table(n_cols, subtbls, cplb)

    if not tab_path.exists():
        tab_path.mkdir(parents=True)
    tab_path.joinpath(f"{meth_name}.tex").write_text(tbl.gen_table(tol=tol))


def _plot_hist(n, res, title, fig_path: Path, bins=20):
    dist = np.random.standard_normal(n)
    fig, ax = plt.subplots()
    n, _, _ = ax.hist(dist, bins=bins, density=True)
    m, *_, M = sorted(res["m"])
    s = max(res["s"])
    y_pnts = [0, np.max(n)]

    ax.plot([m, m], y_pnts, "o-", label="min $\\mu$")
    ax.plot([M, M], y_pnts, "o-", label="max $\\mu$")
    ax.plot([m-s, m-s], y_pnts, "o-", label="min $\\mu$ - max $\\sigma$")
    ax.plot([M+s, M+s], y_pnts, "o-", label="max $\\mu$ + max $\\sigma$")
    ax.legend()
    ax.set_xlabel("x")
    ax.set_ylabel("density")
    plt.title(title)
    fig.show()

    if not fig_path.exists():
        fig_path.mkdir(parents=True)
    fig.savefig(fig_path.joinpath(title))


def lab8(nums, tab_path: Path, fig_path: Path, a=0.05):
    meths = {"norm": caplab("Доверительные интервалы для параметров нормального распределения", "tab:interv_simple"),
             "as": caplab("Доверительные интервалы для параметров произвольного распределения. Асимптотический подход",
                          "tab:interv_asimpt")}
    for k, v in meths.items():
        _gen_table(nums, a, k, v, tab_path)
    for n in nums:
        r = defaultdict(list)
        for k, let in zip(meths, "ms"):
            m_res = eval(f"{k}_{let}({n}, {a})")
            r[let] += [*m_res]
        _plot_hist(n, r, f"Normal(0,1)\nn={n}", fig_path)


if __name__ == "__main__":
    lab8([20, 100], Path("tab/lab8"), Path("figs/lab8"))
