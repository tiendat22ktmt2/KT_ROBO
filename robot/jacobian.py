import numpy as np
from robot.model import Planar3R


def geometric_jacobian(robot: Planar3R, q):

    q1, q2, q3 = np.asarray(q, dtype=float)

    a1 = q1
    a2 = q1 + q2
    a3 = q1 + q2 + q3

    L1 = robot.L1
    L2 = robot.L2
    L3 = robot.L3

    J = np.array([

        [
            -L1*np.sin(a1)
            -L2*np.sin(a2)
            -L3*np.sin(a3),

            -L2*np.sin(a2)
            -L3*np.sin(a3),

            -L3*np.sin(a3)
        ],

        [
            L1*np.cos(a1)
            +L2*np.cos(a2)
            +L3*np.cos(a3),

            L2*np.cos(a2)
            +L3*np.cos(a3),

            L3*np.cos(a3)
        ],

        [
            1.0,
            1.0,
            1.0
        ]

    ], dtype=float)

    return J


def finite_difference_jacobian(
    robot: Planar3R,
    q,
    eps=1e-7
):

    q = np.asarray(q, dtype=float)

    J = np.zeros((3, 3))

    for i in range(3):

        dq = np.zeros(3)
        dq[i] = eps

        plus = robot.fk(q + dq)
        minus = robot.fk(q - dq)

        J[:2, i] = (
            plus[:2] - minus[:2]
        ) / (2 * eps)

        J[2, i] = 1.0

    return J