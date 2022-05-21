from collections import namedtuple
from numpy.random import standard_normal
from pathlib import Path
from scipy.stats import chi2, norm, laplace, uniform
import numpy as np

from lab5_8.table_utils import caplab, SubTable, Table


def _max_plausibility(distr, k, p, tol=2):
    print(f"mu: {np.mean(distr):.{tol}f}, sigma: {np.std(distr):.{tol}f}")

    lims = np.linspace(-2, 2, num=k - 1)
    print(f"chi_2: {chi2.ppf(p, k - 1)}")
    return lims


def _get_data(distr, lims):
    cdfs = list(map(norm.cdf, lims))
    ps = [cur - prev for prev, cur in zip([0] + cdfs, cdfs + [1])]
    ns = list(map(len, [distr[distr < lims[0]]] + \
                  [distr[np.logical_and(lower <= distr, distr < upper)] for lower, upper in zip(lims, lims[1:])] + \
                  [distr[distr >= lims[~0]]]))
    n = len(distr)
    xi_b = [(_n - n * _p) ** 2 / (n * _p) for _n, _p in zip(ns, ps)]
    return xi_b, ps, ns


def _gen_table(lims, ns, ps, xi_b, title, cplb, tab_path: Path, tol=2):
    n = sum(ns)
    lims = ["-\\infty"] + lims.tolist() + ["\\infty"]
    np = [n * _p for _p in ps]
    nnp = [_n - _np for _n, _np in zip(ns, np)]

    data_cols = (ns, ps, np, nnp, xi_b)
    cols = list(zip(range(n), zip(lims, lims[1:]), *data_cols))

    hdrs = "i, \\Centerstack[c]{$borders$\\\\$a_{i-1};\\; a_{i}$}, n_i, p_i, np_i, n_i - np_i, " \
           "\\frac{(n_i - np_i)^2}{np_i}".split(", ")
    footer = ["\\Sigma", "\\text{---}", *map(sum, data_cols)]
    tbl = Table(len(hdrs), SubTable(hdrs, cols, footer), cplb)

    if not tab_path.exists():
        tab_path.mkdir(parents=True)
    tab_path.joinpath(f"{title}.tex").write_text(tbl.gen_table(tol))


def _lab7(distr, a, title, cplb, tab_path: Path):
    k = int(1 + 3.3 * np.log10(len(distr)) + .5)
    p = 1 - a
    lims = _max_plausibility(distr, k, p)
    xi_b, ps, ns = _get_data(distr, lims)
    _gen_table(lims, ns, ps, xi_b, title, cplb, tab_path)


def run_lab7(nums, a, tab_path: Path):
    Data = namedtuple("Data", ["caplab", "distr"])
    distrs = {
        "normal": Data(
            caplab(
                r"Вычисление $\chi^{2}_{B}$ при проверке гипотезы $H_{0}$ "
                r"о нормальном законе распределения $N(x,\hat{\mu}, \hat{\sigma})$",
                "tab:normal_chi_2"
            ),
            standard_normal(nums[0])
        ),
        "laplace": Data(
            caplab(
                r"Вычисление $\chi^{2}_{B}$ при проверке гипотезы $H_{0}$ "
                r"о законе распределения $L(x,\hat{\mu}, \hat{\sigma})$, $n=20$",
                "tab:laplace_chi_2)"
            ),
            laplace.rvs(size=nums[1], scale=1 / np.sqrt(2), loc=0)
        ),
        "uniform": Data(
            caplab(
                r"Вычисление $\chi^{2}_{B}$ при проверке гипотезы $H_{0}$ "
                r"о законе распределения $U(x,\hat{\mu}, \hat{\sigma})$, $n=20$",
                "tab:uniform_chi_2)"
            ),
            uniform.rvs(size=nums[1], scale=2 * np.sqrt(3), loc=-np.sqrt(3))
        )
    }

    for k, v in distrs.items():
        _lab7(v.distr, a, k, v.caplab, tab_path)


if __name__ == "__main__":
    nums = [100, 20]
    a = 0.05
    tab_path = Path("tab/lab7")

    run_lab7(nums, a, tab_path)
