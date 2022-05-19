from pathlib import Path

from lab5_8.lab5 import lab5
from lab5_8.lab6 import lab6
from lab5_8.lab7 import run_lab7
from lab5_8.lab8 import lab8


def main(tab_dir: Path, fig_dir: Path):
    lab5((20, 60, 100), (0, 0.5, 0.9), tab_dir.joinpath("lab5"), fig_dir.joinpath("lab5"))
    lab6((-1.8, 2, 0.2), fig_dir.joinpath("lab6"))
    run_lab7((100, 20), 0.05, tab_dir.joinpath("lab7"))
    lab8((20, 100), tab_dir.joinpath("lab8"))


if __name__ == "__main__":
    root_dir = Path(".")
    tab_dir = root_dir.joinpath("tab")
    fig_dir = root_dir.joinpath("figs")

    main(tab_dir, fig_dir)
