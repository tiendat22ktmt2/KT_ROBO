import numpy as np

from robot.model import Planar3R

from robot.jacobian import (
    geometric_jacobian,
    finite_difference_jacobian
)


def test_jacobian_against_finite_difference():

    robot = Planar3R()

    q = np.array([
        0.3,
        -0.5,
        0.2
    ])

    J_analytic = geometric_jacobian(
        robot,
        q
    )

    J_numeric = finite_difference_jacobian(
        robot,
        q
    )

    assert np.allclose(
        J_analytic,
        J_numeric,
        atol=1e-5
    )