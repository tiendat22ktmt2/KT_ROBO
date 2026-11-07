from robot.model import Planar3R

from trajectories.scenarios import (
    make_scenario,
    make_pose_targets,
    target_feasibility
)


def test_unreachable_scenario():

    robot = Planar3R()

    xy, metadata = make_scenario(
        "unreachable",
        robot,
        samples=20
    )

    targets = make_pose_targets(
        xy,
        orientation=0.0
    )

    feasible = target_feasibility(
        robot,
        targets
    )

    assert metadata["feasible_expected"] is False

    assert not feasible.any()

    assert (xy[:, 0] > robot.max_reach).all()