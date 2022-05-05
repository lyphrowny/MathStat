from pathlib import Path

from lab1 import lab1
from lab2 import lab2
from lab3 import lab3
from setup import set_up


def main():
    distrs = set_up()
    img_dir = Path("lab1_4/imgs")
    table_dir = Path("lab1_4/tables")

    lab1(distrs, [10, 50, 1000], path=img_dir.joinpath("lab1"))
    lab2(distrs, [10, 100, 1000], dir=table_dir.joinpath("lab2"))
    lab3(distrs, [20, 100], plot_dir=img_dir.joinpath("lab3"), table_dir=table_dir.joinpath("lab3"))


if __name__ == "__main__":
    main()
