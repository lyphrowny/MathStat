from collections import defaultdict
from pathlib import Path
from typing import List

import numpy as np

from setup import set_up


def _mean(xs):
    return sum(xs) / len(xs)


def _median(xs):
    # -1 since indexing starts with 0
    n = len(xs) // 2 - 1
    # the oddity condition is reversed due to `-1` above
    return [xs[n + 1], sum(xs[n:n + 2]) / 2][n & 1]


def _z_r(xs):
    return (xs[0] + xs[~0]) / 2


def _z_p(xs, p):
    # -1 due to indexing start from 0
    # dot besides 1 is to convert `np` to float, as this type
    # has `.is_integer()` method
    np = len(xs) * p - 1.
    return xs[int(np) + (not np.is_integer())]


def _z_q(xs, ps):
    return sum(_z_p(xs, p) for p in ps) / len(ps)


def _tr_med(xs, r=0.25):
    n = len(xs)
    nr = int(n * r)
    # substitute 1 as indexes start from 0
    return 1 / (n - 2 * nr) * sum(xs[nr:n - nr])


def _make_table_part(distr, ps_num, char_names, chars):
    n = len(char_names)
    tol = 6
    pre = f"{distr.__class__.__name__} n={ps_num}{' & ' * n}\\\\\n" \
          f"\\hline \n" \
          f"& {' & '.join(f'${name}$' for name in char_names)} \\\\\n" \
          f"\\hline \n"
    mid = ''.join(f"${name}$ & {' & '.join(f'{ch:.{tol}f}' for ch in char)} \\\\\n" for name, char in chars.items())
    suf = f"\\hline\n" \
          f"\\multicolumn{{{n+1}}}{{c}}{{}} \\\\\n"
    return [pre, mid, suf]


def _make_table(table: List[str], cols_num):
    return f"""\\begin{{table}}[H]
    \\centering
    \\begin{{tabular}}{{{"|".join("c" * (cols_num+1))}}}
{"".join(table[:~0])}
    \\end{{tabular}}
    \\caption{{}}
    \\label{{}}
\\end{{table}}"""


def lab2(distrs, ps_num, table_dir, times=1000):
    if not table_dir.exists():
        table_dir.mkdir(parents=True)
    for distr in distrs:
        dest = table_dir.joinpath(f"{distr.__class__.__name__}.tex")
        dest.write_text(_gen_table(distr, ps_num, times))


def _gen_table(distr, ps_num, times=1000):
    d = defaultdict(list)
    table = []
    for p_num in ps_num:
        for _ in range(times):
            x = sorted(distr.get_rvs(p_num))
            d[r"\bar{x}"].append(_mean(x))
            d["med\\; x"].append(_median(x))
            d["z_R"].append(_z_r(x))
            d["z_Q"].append(_z_q(x, [0.25, 0.75]))
            d["z_{tr}"].append(_tr_med(x))
        chars = defaultdict(list)
        for v in d.values():
            chars["E(z)"].append(_mean(v))
            chars["D(z)"].append(_mean(np.power(v, 2)) - np.power(chars["E(z)"][~0], 2))
            chars["E(z) - \\sqrt{D(z)}"].append(chars["E(z)"][~0] - np.sqrt(chars["D(z)"][~0]))
            chars["E(z) + \\sqrt{D(z)}"].append(chars["E(z)"][~0] + np.sqrt(chars["D(z)"][~0]))
        table.extend(_make_table_part(distr, p_num, d.keys(), chars))
    return _make_table(table, len(d.keys()))


if __name__ == "__main__":
    distr = set_up()[0]
    test_data = np.sort(distr.get_rvs(1000))
    # test_data = list(range(1, 11))
    quarts = [0.25, 0.75]
    print("my quartiles")
    print(", ".join(f"x[n*{p}]: {_z_p(test_data, p)}" for p in quarts))
    print(f"z_Q: {_z_q(test_data, quarts)}")
    print()
    print("numpy quartiles")
    print(", ".join(f"x[n*{p}]: {np.quantile(test_data, p)}" for p in quarts))
    print(f"z_Q: {sum(np.quantile(test_data, p) for p in quarts) / 2}")
    print()
    print(f"z_tr: {_tr_med(test_data)}")
    # distrs = set_up()
    # lab2(distrs, [10, 100, 1000], Path("tables"))
