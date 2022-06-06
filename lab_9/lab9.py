import csv
from collections import namedtuple
from itertools import starmap
from operator import attrgetter
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize

Octave = namedtuple("Octave", "a b ws")


def _read_csv(file_name):
    # turn each line's col into float; unpack the list of tuples and feed these tuples
    # to zip, for it to gather them element-wise (by columns)
    return zip(*[map(float, line) for line in csv.reader(open(file_name), delimiter=";")])


def _plot_datas(ds, title, subttls, fig_dir: Path):
    fig, axs = plt.subplots(1, len(ds), figsize=(10.5, 4), tight_layout=True)
    fig.suptitle(title)
    for d, ax, subttl in zip(ds, axs, subttls):
        x, y = zip(*enumerate(d, start=1))
        ax.plot(x, y, label=subttl)
        ax.set(title=subttl, xlabel="n", ylabel="mV")
        ax.legend()
    fig.show()
    fig.savefig(fig_dir.joinpath(title))


def _plot_dintervals(ds, title, subttls, fig_dir: Path, tol=1e-4):
    fig, axs = plt.subplots(1, len(ds), figsize=(10.5, 4), tight_layout=True)
    fig.suptitle(title)
    for d, ax, subttl in zip(ds, axs, subttls):
        ax.vlines(1, d[0] - tol, d[0] + tol, label=subttl)
        for n, v in enumerate(d[1:], start=2):
            ax.vlines(n, v - tol, v + tol)
        ax.set(title=subttl, xlabel="n", ylabel="mV")
        ax.legend()
    fig.show()
    fig.savefig(fig_dir.joinpath(title))


def _read_octave(file):
    with open(file, "r") as f:
        a, b = map(float, f.readline().split())
        ws = np.array([*map(float, f.readlines())])
    return a, b, ws


def _lin_drift(ds, octs: Octave, title, subttls, fig_dir: Path, tol=1e-4):
    fig, axs = plt.subplots(1, len(ds), figsize=(10.5, 4), tight_layout=True)
    fig.suptitle(title)
    for d, oct, ax, subttl in zip(ds, octs, axs, subttls):
        d, ws = map(np.array, (d, oct.ws))
        xs = np.arange(1, len(d) + 1)
        ax.vlines(xs, d - ws * tol, d + ws * tol, label="I")
        ax.plot(xs, oct.a + xs * oct.b, label="Lin", color="orange", linewidth=2.5)
        ax.set(title=subttl, xlabel="n", ylabel="mV")
        ax.legend()
    fig.show()
    fig.savefig(fig_dir.joinpath(title))


def _plot_whist(octs, title, subttls, fig_dir: Path):
    fig, axs = plt.subplots(1, len(octs), figsize=(10.5, 4), tight_layout=True)
    fig.suptitle(title)
    for oct, ax, subttl in zip(octs, axs, subttls):
        ax.hist(oct.ws, label="w")
        ax.set(title=subttl, xlabel="weight", ylabel="n")
        ax.legend()
    fig.show()
    fig.savefig(fig_dir.joinpath(title))


def _plot_no_drift(ds, octs: Octave, title, subttls, fig_dir: Path, tol=1e-4):
    fig, axs = plt.subplots(1, len(ds), figsize=(10.5, 4), tight_layout=True)
    fig.suptitle(title)
    for d, oct, ax, subttl in zip(ds, octs, axs, subttls):
        d, ws = map(np.array, (d, oct.ws))
        xs = np.arange(1, len(d) + 1)
        fixed = d - xs * oct.b
        ax.vlines(xs, fixed - ws * tol, fixed + ws * tol, label="I")
        ax.plot(xs, np.full(xs[~0], oct.a), label="Lin", color="orange", linewidth=2.5)
        ax.set(title=subttl, xlabel="n", ylabel="mV")
        ax.legend()
    fig.show()
    fig.savefig(fig_dir.joinpath(title))


def _plot_ndhist(ds, octs, title, subttls, fig_dir: Path):
    fig, axs = plt.subplots(1, len(ds), figsize=(10.5, 4), tight_layout=True)
    fig.suptitle(title)
    for d, oct, ax, subttl in zip(ds, octs, axs, subttls):
        d, ws = map(np.array, (d, oct.ws))
        xs = np.arange(1, len(d) + 1)
        fixed = d - xs * oct.b
        ax.hist(fixed, label="$I^c$")
        ax.set(title=subttl, xlabel="weight", ylabel="n")
        ax.legend()
    fig.show()
    fig.savefig(fig_dir.joinpath(title))


def _plot_jakkar(ds, octs, title, fig_dir: Path, tol=1e-4):
    *ds, = map(np.array, ds)
    xs = np.arange(1, len(ds[0]) + 1)
    fix1, fix2 = [np.array([d - tol * oct.ws, d + tol * oct.ws]) - oct.b * xs for d, oct in zip(ds, octs)]

    # rint_lower, rint_upper = (1.04, 1.07)
    rint_lower, rint_upper = (0.5, 1.5)
    rint = np.linspace(rint_lower, rint_upper, 10000)

    def jac(r):
        d_new = np.hstack((fix1 * r, fix2))
        min_lower, *_, max_lower = np.sort(d_new[0, :])
        min_upper, *_, max_upper = np.sort(d_new[1, :])
        return (min_upper - max_lower) / (max_upper - min_lower)

    _ctol = 1e-11
    opt_r = scipy.optimize.fmin(lambda x: -jac(x), (rint_upper + rint_lower) / 2, xtol=_ctol)
    ma = scipy.optimize.root(jac, rint_lower, method="lm", tol=_ctol).x
    mi = scipy.optimize.root(jac, rint_upper, method="lm", tol=_ctol).x

    plt.title(title)
    plt.plot(rint, np.vectorize(jac)(rint), label="Jaccard", zorder=1)
    _prec = 16
    for v, plh in zip((opt_r, mi, ma), ("$R_{opt}=%.*f$", "$min=%.*f$", "$max=%.*f$")):
        plt.scatter(v, jac(v), label=plh % (_prec, v), zorder=2)
    plt.xlabel("$R_{21}$")
    plt.ylabel("Jaccard")
    plt.legend()
    plt.show()
    plt.savefig(fig_dir.joinpath(title))

    return opt_r


def _plot_jhist(ds, octs, r_opt, title, fig_dir: Path, tol=1e-4):
    *ds, = map(np.array, ds)
    xs = np.arange(1, len(ds[0]) + 1)
    fix1, fix2 = [np.array([d - tol * oct.ws, d + tol * oct.ws]) - oct.b * xs for d, oct in zip(ds, octs)]
    d_new = np.hstack((fix1 * r_opt, fix2))

    plt.title(title)
    plt.hist(np.sum(d_new, axis=0) / 2, label="combined with $R_{opt}$")
    plt.xlabel("weights")
    plt.ylabel("n")
    plt.legend()
    plt.show()
    plt.savefig(fig_dir.joinpath(title))


def lab9(data_dir: Path, fig_dir: Path, tol=1e-4):
    if not fig_dir.exists():
        fig_dir.mkdir(parents=True)
    *file_names, = data_dir.glob("*.csv")
    ds, es = zip(*map(_read_csv, file_names))
    *subttls, = map(attrgetter("stem"), file_names)
    # _plot_datas(ds, "Experiment data", file_names, fig_dir)
    # _plot_dintervals(ds, "Intervaled data", file_names, fig_dir, tol=tol)

    *octs, = starmap(Octave, map(_read_octave, data_dir.glob("*.txt")))
    # _lin_drift(ds, octs, "Drifted data", subttls, fig_dir, tol=tol)
    # _plot_whist(octs, "Weights' histogram", subttls, fig_dir)
    # _plot_no_drift(ds, octs, "Data w\\o drift", subttls, fig_dir, tol=tol)
    # _plot_ndhist(ds, octs, "$I^c$ histogram", subttls, fig_dir)
    r_opt = _plot_jakkar(ds, octs, "Jaccard vs $R_{21}$", fig_dir, tol=tol)
    _plot_jhist(ds, octs, r_opt, "Histogram of combined data with $R_{opt}$", fig_dir, tol=tol)


if __name__ == "__main__":
    lab9(Path("./data"), Path("./imgs"))
