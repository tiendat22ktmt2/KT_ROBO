import numpy as np


def time_derivatives(
    t,
    q
):

    t = np.asarray(
        t,
        dtype=float
    )

    q = np.unwrap(
        np.asarray(q, dtype=float),
        axis=0
    )

    if len(t) < 4:
        raise ValueError(
            "At least 4 samples are required "
            "to estimate velocity, acceleration and jerk."
        )

    velocity = np.gradient(
        q,
        t,
        axis=0,
        edge_order=2
    )

    acceleration = np.gradient(
        velocity,
        t,
        axis=0,
        edge_order=2
    )

    jerk = np.gradient(
        acceleration,
        t,
        axis=0,
        edge_order=2
    )

    return (
        velocity,
        acceleration,
        jerk
    )