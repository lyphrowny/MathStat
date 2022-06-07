from collections import namedtuple
from contextlib import contextmanager
from itertools import starmap
from operator import attrgetter
from pathlib import Path
import csv
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize

Octave = namedtuple("Octave", "a b ws")
# default tolerance
dtol = 1e-4


class Plotter:
    def __init__(self, shape, subttls, fig_dir: Path, need_show=False):
        self.__n_subplots, _ = shape
        self._subttls = subttls
        self._dir = fig_dir
        self._need_show = need_show

    def __plot(self, title, x_label="n", y_label="mV"):
        fig, axs = plt.subplots(1, self.__n_subplots, figsize=(10.5, 4), tight_layout=True)
        fig.suptitle(title)
        for ax, subttl in zip(axs, self._subttls):
            yield ax
            ax.set(title=subttl, xlabel=x_label, ylabel=y_label)
            ax.legend()
        if self._need_show:
            fig.show()
        fig.savefig(self._dir.joinpath(title))
        plt.close()

    @contextmanager
    def acq(self, title, x_label="n", y_label="mV"):
        # prepare the generator
        axs = self.__plot(title, x_label=x_label, y_label=y_label)
        try:
            # return the generator in order to iterate over it
            yield axs
        finally:
            # make the finishing touches, i.e. run the generator func
            # till its end
            # no need to raise StopIteration, hence None
            next(axs, None)


def _read_csv(file_name):
    # the second column isn't used in the whole program, let's ignore it altogether
    return [float(mv) for (mv, e) in csv.reader(open(file_name), delimiter=";")]


def _plot_data(ds, enumed, title, lbls):
    with P.acq(title) as axs:
        for d, ax, lbl in zip(ds, axs, lbls):
            ax.plot(enumed, d, label=lbl)


def _plot_intr(ds, enumed, title, lbls, tol=dtol):
    with P.acq(title) as axs:
        for d, ax, lbl in zip(ds, axs, lbls):
            ax.vlines(enumed, d - tol, d + tol, label=lbl)


def _read_octave(file):
    with open(file, "r") as f:
        a, b = map(float, f.readline().split())
        ws = np.array([*map(float, f.readlines())])
    return a, b, ws


def _plot_lin_drift(ds, enumed, octs: Octave, title, tol=dtol):
    w_intrs = []
    with P.acq(title) as axs:
        for d, oct, ax in zip(ds, octs, axs):
            w_intrs.append(np.array([d - oct.ws * tol, d + oct.ws * tol]))
            ax.vlines(enumed, *w_intrs[~0], label="I")
            ax.plot(enumed, oct.a + enumed * oct.b, label="Lin", color="orange", linewidth=2.5)
    return np.array(w_intrs)


def _plot_whist(octs, title):
    with P.acq(title, x_label="weight", y_label="n") as axs:
        for oct, ax in zip(octs, axs):
            ax.hist(oct.ws, label="w")


def _plot_no_drift(w_intrs, enumed, octs: Octave, title):
    f_intrs = []
    with P.acq(title) as axs:
        for w_intr, oct, ax in zip(w_intrs, octs, axs):
            f_intrs.append(w_intr - enumed * oct.b)
            ax.vlines(enumed, *f_intrs[~0], label="I")
            ax.plot(enumed, np.full(enumed[~0], oct.a), label="Lin", color="orange", linewidth=2.5)
    return np.array(f_intrs)


def _plot_ndhist(f_intrs, title):
    with P.acq(title, x_label="mV", y_label="n") as axs:
        for (lo, up), ax, in zip(f_intrs, axs):
            ax.hist((lo + up) / 2, label="$I^c$")


def _plot_jaccard(f_intrs, title, fig_dir: Path, need_show=False):
    # rint_lower, rint_upper = (1.04, 1.07)
    rint_lower, rint_upper = (0.5, 1.5)
    rint = np.linspace(rint_lower, rint_upper, 10000)

    def jac(r):
        d_new = np.hstack((f_intrs[0, ...] * r, f_intrs[1, ...]))
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
    if need_show:
        plt.show()
    plt.savefig(fig_dir.joinpath(title))
    plt.close()

    return opt_r


def _plot_jhist(f_intrs, r_opt, title, fig_dir: Path, need_show=False):
    d_new = np.hstack((f_intrs[0, ...] * r_opt, f_intrs[1, ...]))

    plt.title(title)
    plt.hist(np.sum(d_new, axis=0) / 2, label="combined with $R_{opt}$")
    plt.xlabel("mV")
    plt.ylabel("n")
    plt.legend()
    if need_show:
        plt.show()
    plt.savefig(fig_dir.joinpath(title))
    plt.close()


def lab9(data_dir: Path, fig_dir: Path, tol=dtol, need_show=False):
    if not fig_dir.exists():
        fig_dir.mkdir(parents=True)
    file_names = sorted(data_dir.glob("*.csv"))
    ds = np.array([*map(_read_csv, file_names)])
    enumed = np.arange(1, ds.shape[1] + 1)

    # plotting-related data
    *subttls, = map(attrgetter("stem"), file_names)
    global P
    P = Plotter(ds.shape, subttls, fig_dir, need_show=need_show)

    _plot_data(ds, enumed, "Experiment data", subttls)
    _plot_intr(ds, enumed, "Intervaled data", subttls, tol=tol)

    *octs, = starmap(Octave, map(_read_octave, sorted(data_dir.glob("*.txt"))))
    # weighted intervals
    w_intrs = _plot_lin_drift(ds, enumed, octs, "Drifted data", tol=tol)
    _plot_whist(octs, "Weights' histogram")
    # fixed intervals
    f_intrs = _plot_no_drift(w_intrs, enumed, octs, "Remove linear drift")
    _plot_ndhist(f_intrs, "Remove linear drift histogram")
    r_opt = _plot_jaccard(f_intrs, "Jaccard vs $R_{21}$", fig_dir, need_show=need_show)
    _plot_jhist(f_intrs, r_opt, "Histogram of combined data with $R_{opt}$", fig_dir, need_show=need_show)


if __name__ == "__main__":
    lab9(Path("./data"), Path("./figs"))
