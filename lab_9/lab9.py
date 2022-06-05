import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


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


def lab9(data_dir: Path, fig_dir: Path, tol=1e-4):
    if not fig_dir.exists():
        fig_dir.mkdir(parents=True)
    *file_names, = data_dir.glob("*.csv")
    ds, es = zip(*map(_read_csv, file_names))
    _plot_datas(ds, "Experiment data", file_names, fig_dir)
    _plot_dintervals(ds, "Intervaled data", file_names, fig_dir, tol=tol)


if __name__ == "__main__":
    lab9(Path("./data"), Path("./imgs"))
