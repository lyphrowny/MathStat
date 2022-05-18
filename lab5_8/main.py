from pathlib import Path

from lab5_8.lab5 import lab5


def main(tab_dir: Path, fig_dir: Path):
    lab5((20, 60, 100), (0, 0.5, 0.9), tab_dir.joinpath("lab5"), fig_dir.joinpath("lab5"))


if __name__ == "__main__":
    root_dir = Path(".")
    tab_dir = root_dir.joinpath("tab")
    fig_dir = root_dir.joinpath("figs")

    main(tab_dir, fig_dir)
