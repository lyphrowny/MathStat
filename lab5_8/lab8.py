from pathlib import Path
from scipy.stats import chi2, t
import numpy as np


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


def _gen_table(nums, a, meth_name, tab_path: Path):
    n_cols = 3
    spacer = " " * 8
    ending = " \\\\\n"
    head = f"""\\begin{{table}}[H]
\\centering
\\begin{{tabular}}{{{'|'.join('c' * n_cols)}}}
"""
    tail = """    \\end{tabular}
    \\caption{}
    \\label{}
\\end{table}"""

    _trunc = lambda v, tol=2: f"{v:.{tol}f}"
    _prettify = lambda lims, name: f"{lims[0]} < {name} < {lims[1]}"

    cntx = []
    for n in nums:
        title = f"n={n}"
        m = _prettify(list(map(_trunc, eval(f"{meth_name}_m({n}, {a})"))), "m")
        s = _prettify(list(map(_trunc, eval(f"{meth_name}_s({n}, {a})"))), "s")
        cntx.extend([spacer + " & ".join((title, "m", "s")) + ending, spacer + " & ".join(("", m, s)) + ending,
                     spacer + "\\hline\n", spacer + f"\\multicolumn{{{n_cols}}}{{c}}{{}}" + ending])

    if not tab_path.exists():
        tab_path.mkdir(parents=True)
    tab_path.joinpath(f"{meth_name}.tex").write_text(head + "".join(cntx) + tail)


def lab8(nums, tab_path: Path, a=0.05):
    _gen_table(nums, a, "norm", tab_path)
    _gen_table(nums, a, "as", tab_path)


if __name__ == "__main__":
    lab8([20, 100], Path("tab/lab8"))
