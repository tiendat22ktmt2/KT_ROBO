import numpy as np

from .shapes import shape_points


def make_scenario(
    name,
    robot,
    samples=401,
    scale=25.0,
    center=(175.0, 0.0),
    orientation=0.0
):

    name = name.lower()

    if name in (
        "normal_circle",
        "normal_square",
        "normal_star"
    ):

        shape = name.replace(
            "normal_",
            ""
        )

        xy = shape_points(
            shape,
            samples,
            scale,
            center
        )

        return xy, {
            "feasible_expected": True,
            "description": (
                f"{shape.title()} in nominal workspace"
            )
        }

    if name == "near_singularity":

        radius = (
            robot.max_reach
            - np.linspace(
                12.0,
                0.05,
                samples
            )
        )

        theta = np.linspace(
            -0.08,
            0.08,
            samples
        )

        xy = np.column_stack([
            radius * np.cos(theta),
            radius * np.sin(theta)
        ])

        return xy, {
            "feasible_expected": True,
            "description": (
                "Targets approach full-extension singularity"
            )
        }

    if name == "workspace_boundary":

        radius = (
            robot.max_reach
            - np.linspace(
                35.0,
                1.0,
                samples
            )
        )

        theta = np.linspace(
            -0.25,
            0.25,
            samples
        )

        xy = np.column_stack([
            radius * np.cos(theta),
            radius * np.sin(theta)
        ])

        return xy, {
            "feasible_expected": True,
            "description": (
                "Targets near outer workspace boundary"
            )
        }

    if name == "unreachable":

        xy = np.zeros(
            (samples, 2)
        )

        xy[:, 0] = (
            robot.max_reach + 10.0
        )

        xy[:, 1] = np.linspace(
            -2.0,
            2.0,
            samples
        )

        return xy, {
            "feasible_expected": False,
            "description": (
                "Targets outside maximum reach"
            )
        }

    raise ValueError(
        f"Unknown scenario: {name}"
    )


def make_pose_targets(
    xy,
    orientation=0.0
):

    xy = np.asarray(
        xy,
        dtype=float
    )

    phi = np.full(
        len(xy),
        orientation
    )

    return np.column_stack([
        xy,
        phi
    ])


def target_is_reachable(
    robot,
    target,
    tolerance=1e-10
):
    """
    Check reachability for a target pose with fixed orientation.

    For a 3R planar robot, remove the third link to obtain
    the wrist position, then check the two-link workspace.
    """

    x, y, phi = np.asarray(
        target,
        dtype=float
    )

    wx = x - robot.L3 * np.cos(phi)
    wy = y - robot.L3 * np.sin(phi)

    wrist_distance_sq = wx**2 + wy**2

    c2 = (
        wrist_distance_sq
        - robot.L1**2
        - robot.L2**2
    ) / (
        2.0 * robot.L1 * robot.L2
    )

    return (
        -1.0 - tolerance
        <= c2
        <= 1.0 + tolerance
    )


def target_feasibility(
    robot,
    targets
):
    """
    Return a Boolean feasibility mask for each target.
    """

    return np.array([
        target_is_reachable(robot, target)
        for target in targets
    ], dtype=bool)