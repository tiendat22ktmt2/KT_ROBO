
import numpy as np
import pytest

from robot.model import Planar3R
from config import SOLVER

from solvers.analytical import solve_analytical
from solvers.pinv import solve_pinv
from solvers.dls import solve_dls
from solvers.adaptive_dls import solve_adaptive_dls
from solvers.lm import solve_lm


NUMERICAL_SOLVERS = [
    solve_pinv,
    solve_dls,
    solve_adaptive_dls,
    solve_lm,
]


ALL_SOLVERS = [
    ("analytical", solve_analytical),
    ("pinv", solve_pinv),
    ("dls", solve_dls),
    ("adaptive_dls", solve_adaptive_dls),
    ("lm", solve_lm),
]


def test_analytical_ik_round_trip():
    robot = Planar3R()

    q_original = np.array([
        0.4,
        0.7,
        -0.3
    ])

    target = robot.fk(q_original)

    result = solve_analytical(
        robot,
        target,
        elbow=1,
        cfg=SOLVER
    )

    assert result.converged

    achieved = robot.fk(result.q)

    assert np.allclose(
        achieved,
        target,
        atol=1e-6
    )


@pytest.mark.parametrize(
    "elbow",
    [1, -1]
)
def test_analytical_both_elbow_configurations(elbow):
    robot = Planar3R()

    q_original = np.array([
        0.4,
        0.7,
        -0.3
    ])

    target = robot.fk(q_original)

    result = solve_analytical(
        robot,
        target,
        elbow=elbow,
        cfg=SOLVER
    )

    assert result.converged

    assert np.allclose(
        robot.fk(result.q),
        target,
        atol=1e-6
    )


@pytest.mark.parametrize(
    "solver",
    NUMERICAL_SOLVERS
)
def test_numerical_ik_round_trip(solver):
    robot = Planar3R()

    q_original = np.array([
        0.4,
        0.7,
        -0.3
    ])

    target = robot.fk(q_original)

    q0 = np.array([
        0.3,
        0.5,
        -0.1
    ])

    result = solver(
        robot,
        target,
        q0,
        SOLVER
    )

    assert result.converged

    assert np.allclose(
        robot.fk(result.q),
        target,
        atol=1e-4
    )


@pytest.mark.parametrize(
    "solver_name, solver",
    ALL_SOLVERS
)
def test_ik_detects_geometrically_unreachable_target(
    solver_name,
    solver
):
    robot = Planar3R()

    # Wrist position is outside the reachable workspace.
    target = np.array([
        300.0,
        0.0,
        0.0
    ])

    q0 = np.zeros(3)

    if solver_name == "analytical":
        result = solver(
            robot,
            target,
            elbow=1,
            q_seed=q0,
            cfg=SOLVER
        )
    else:
        result = solver(
            robot,
            target,
            q0,
            SOLVER
        )

    assert not result.converged

    assert result.reason == (
        "geometrically_unreachable"
    )

    assert result.iterations == 0
