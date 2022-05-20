from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms


def _get_pearson(pnts):
    x, y = list(zip(*pnts))
    x_mean, y_mean = list(map(np.mean, (x, y)))

    h, l_x, l_y = 0, 0, 0
    for _x, _y in pnts:
        d_x, d_y = _x - x_mean, _y - y_mean
        h += d_x * d_y
        l_x += d_x ** 2
        l_y += d_y ** 2
    return h / np.sqrt(l_x * l_y)


def _get_quad_coeff(pnts):
    quads = [0] * 4

    for x, y in pnts:
        # 1  |  0
        # -------
        # ~1 | ~0
        quads[eval(f"{'~' * (bool(y < 0))}+{bool(x < 0)}")] += 1
    return (sum(quads[::2]) - sum(quads[1::2])) / len(pnts)


def _get_spearman(pnts):
    n = len(pnts)
    mean = (n + 1) / 2
    rank_x, rank_y = list(map(lambda xs: {x: r for r, x in enumerate(np.sort(xs))}, zip(*pnts)))

    h, l_x, l_y = 0, 0, 0
    for _x, _y in pnts:
        dr_x, dr_y = rank_x[_x] - mean, rank_y[_y] - mean
        h += dr_x * dr_y
        l_x += dr_x ** 2
        l_y += dr_y ** 2
    return h / np.sqrt(l_x * l_y)


def _conf_ellipse(x, y, ax, n_std=3.0, **kwargs):
    pearson = _get_pearson(list(zip(x, y)))
    r_x, r_y = np.sqrt(1 + pearson), np.sqrt(1 - pearson)
    cov = np.cov(x, y)
    ell = Ellipse((0, 0), width=r_x * 2, height=r_y * 2, facecolor="none", **kwargs)
    trans = transforms.Affine2D().rotate_deg(45).scale(np.sqrt(cov[0, 0]) * n_std,
                                                       np.sqrt(cov[1, 1]) * n_std).translate(np.mean(x), np.mean(y))
    ell.set_transform(trans + ax.transData)
    return ax.add_patch(ell)


def _plot_ellipses(distrs, title, subtitles, dir_path: Path):
    fig, axs = plt.subplots(1, len(distrs), figsize=(10.5, 4), tight_layout=True)
    fig.suptitle(title)
    for ax, distr, subttl in zip(axs, distrs, subtitles):
        x, y = list(zip(*distr()))
        ax.scatter(x, y)
        _conf_ellipse(x, y, ax, edgecolor='red')
        ax.set_title(subttl)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
    fig.show()

    if not dir_path.exists():
        dir_path.mkdir(parents=True)
    fig.savefig(dir_path.joinpath(title))


def _make_subtable(data, title: str):
    c = defaultdict(list)
    for pnts in data:
        c["r"].append(_get_pearson(pnts))
        c["r_{S}"].append(_get_quad_coeff(pnts))
        c["r_{Q}"].append(_get_spearman(pnts))
    r = defaultdict(list)
    for v in c.values():
        r["E(z)"].append(np.mean(v))
        r["E(z^2)"].append(np.mean([_v * _v for _v in v]))
        r["D(z)"].append(np.std(v))

    beautify = lambda s: f"${s}$"
    tol = 6
    trunc = lambda v: f"{v:.{tol}f}"
    head = " & ".join(map(beautify, (title, *c.keys()))) + " \\\\"
    conts = [f"{' & '.join((beautify(k), *map(trunc, v)))} \\\\" for k, v in r.items()]
    return [head] + conts


def _gen_table(table):
    n = table[0][0].count('&') + 1
    spacer = " " * 8
    filler = f"\n{spacer}\\hline\n" \
             f"{spacer}\\multicolumn{{{n}}}{{c}}{{}} \\\\\n"
    tab = f"""\\begin{{table}}[H]
    \\centering
    \\begin{{tabular}}{{{'|'.join('c' * n)}}}
{filler.join(chr(10).join(spacer + line for line in subtbl) for subtbl in table)}
    \\end{{tabular}}
    \\caption{{}}
    \\label{{}}
\\end{{table}}"""
    return tab


def _make_table(distrs, title, subtitles, tab_path: Path, times=1000):
    table = []
    for subttl, distr in zip(subtitles, distrs):
        data = [distr() for _ in range(times)]
        table.append(_make_subtable(data, subttl))

    if not tab_path.exists():
        tab_path.mkdir(parents=True)
    tab_path.joinpath(f"{title}.tex").write_text(_gen_table(table))


def __get_cov(rho, sigma=1):
    return [[sigma, rho], [rho, sigma]]


def lab5(nums, rhos, tab_path: Path, fig_path: Path):
    rng = np.random.default_rng()

    covs = tuple(map(__get_cov, rhos))
    uber_distrs = []
    for n in nums:
        distrs = []
        for c in covs:
            distrs.append(lambda d=rng.multivariate_normal(mean=[0, 0], cov=c, size=n): d)
        title = f"n={n}"
        subttls = [f"$\\rho={rho}$" for rho in rhos]
        _make_table(distrs, title, subttls, tab_path)
        _plot_ellipses(distrs, title, subttls, fig_path)

        uber_distrs.append(lambda d=0.9 * rng.multivariate_normal(mean=[0, 0], cov=__get_cov(0.9), size=n) + \
                                    0.1 * rng.multivariate_normal(mean=[0, 0], cov=__get_cov(-0.9, 10), size=n): d)
    title = f"mix"
    subttls = [f"$n={n}$" for n in nums]
    _make_table(uber_distrs, title, subttls, tab_path)
    _plot_ellipses(uber_distrs, title, subttls, fig_path)


if __name__ == "__main__":
    lab5((20, 60, 100), (0, 0.5, 0.9), Path("tab/lab5"), Path("figs/lab5"))
