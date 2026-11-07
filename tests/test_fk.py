import numpy as np

from robot.model import Planar3R


def test_zero_configuration():

    robot = Planar3R(
        L1=100.0,
        L2=80.0,
        L3=40.0
    )

    q = np.array([
        0.0,
        0.0,
        0.0
    ])

    pose = robot.fk(q)

    expected = np.array([
        220.0,
        0.0,
        0.0
    ])

    assert np.allclose(
        pose,
        expected,
        atol=1e-9
    )