from numpy.random import standard_normal
from pathlib import Path
from scipy.stats import chi2, norm, laplace, uniform
import numpy as np


def _max_plausibility(distr, k, p):
    tol = 2
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


def _gen_table(lims, ns, ps, xi_b, title, tab_path: Path):
    n = sum(ns)
    tol = 2
    _trunc = lambda s: f"{s:.{tol}f}" if type(s) != tuple and type(s) != int else f"{s}"

    lims = ["$-\\infty$"] + list(map(_trunc, lims)) + ["$\\infty$"]
    np = [n * _p for _p in ps]
    nnp = [_n - _np for _n, _np in zip(ns, np)]

    cols = list(zip(range(n), zip(lims[::2], lims[1::2]), ns, ps, np, nnp, xi_b))
    spacer = " " * 8
    ending = " \\\\\n"
    head = f"""\\begin{{table}}[H]
    \\centering
    \\begin{{tabular}}{{{'|'.join('c' * len(cols[0]))}}}
"""
    headers = spacer + " & ".join(map(lambda s: f"${s}$",
                                      "i, borders a_{i-1} a_{i}, n_i, p_i, np_i, n_i - np_i, "
                                      "\\frac{(n_i - np_i)^2}{np_i}".split(", "))) + ending
    conts = ending.join(spacer + " & ".join(map(_trunc, col)) for col in cols) + ending
    last = spacer + " & ".join(("$\Sigma$", "--", *map(_trunc, (sum(v) for v in (ns, ps, np, nnp, xi_b))))) + " \n"
    tail = """    \\end{tabular}
    \\caption{}
    \\label{}
\\end{table}"""

    if not tab_path.exists():
        tab_path.mkdir(parents=True)
    tab_path.joinpath(f"{title}.tex").write_text(head + headers + conts + last + tail)


def _lab7(distr, a, title, tab_path: Path):
    k = int(1 + 3.3 * np.log(len(distr)))
    p = 1 - a
    lims = _max_plausibility(distr, k, p)
    xi_b, ps, ns = _get_data(distr, lims)
    _gen_table(lims, ns, ps, xi_b, title, tab_path)


def run_lab7(nums, a, tab_path: Path):
    a = 0.05
    distr = standard_normal(nums[0])
    _lab7(distr, a, "normal", tab_path)

    num = nums[1]
    for ttl, d in zip("laplace uniform".split(), (
            laplace.rvs(size=num, scale=1 / np.sqrt(2), loc=0),
            uniform.rvs(size=num, scale=2 * np.sqrt(3), loc=-np.sqrt(3)))):
        _lab7(d, a, ttl, tab_path)


if __name__ == "__main__":
    num = 100
    a = 0.05
    distr = standard_normal(num)
    _lab7(distr, a, "normal", Path("tab/lab7"))

    num = 20
    for ttl, d in zip("laplace uniform".split(), (
            laplace.rvs(size=num, scale=1 / np.sqrt(2), loc=0),
            uniform.rvs(size=num, scale=2 * np.sqrt(3), loc=-np.sqrt(3)))):
        _lab7(d, a, ttl, Path("tab/lab7"))
