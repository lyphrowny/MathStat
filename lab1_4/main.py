from pathlib import Path

from lab1 import lab1
from setup import set_up


def main():
    distrs = set_up()
    img_dir = Path("lab1_4/imgs")

    lab1(distrs, [10, 50, 1000], path=img_dir.joinpath("lab1"))


if __name__ == "__main__":
    main()
