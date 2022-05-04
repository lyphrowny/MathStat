from pathlib import Path
import matplotlib.pyplot as plt

from setup import set_up

__all__ = ["lab1"]


def lab1(distrs, ps_num, bins_num=20, path=Path("../imgs")):

    if not path.exists():
        path.mkdir(parents=True)

    for d in distrs:
        # tight layouts make the plot less cramped
        fig, axs = plt.subplots(1, len(ps_num), figsize=(10.5, 4), tight_layout=True)
        title = f"{d.__class__.__name__}"
        fig.suptitle(title)
        for p_num, ax in zip(ps_num, axs):
            data = d.get_rvs(p_num)
            # `bins` will have the edges of the built bins
            _, bins, _ = ax.hist(data, bins_num, density=True)
            ax.plot(bins, list(map(d.get_pdf, bins)))
            ax.set_title(f"n={p_num}")
        plt.savefig(path.joinpath(title))
        plt.close(fig)


if __name__ == "__main__":
    distrs = set_up()
    bins_num = 20
    points_num = [10, 50, 1000]
    lab1(distrs, points_num, bins_num, Path("imgs/lab1.1"))
