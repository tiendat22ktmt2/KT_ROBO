from config import (
    ROBOT,
    SOLVER,
    EXPERIMENT
)

from robot.model import Planar3R

from experiments.runner import run_all


def main():

    robot = Planar3R(
        ROBOT.L1,
        ROBOT.L2,
        ROBOT.L3
    )

    run_all(
        robot,
        SOLVER,
        EXPERIMENT
    )


if __name__ == "__main__":
    main()