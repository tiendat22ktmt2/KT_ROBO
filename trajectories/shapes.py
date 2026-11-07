import numpy as np


def resample_closed_polygon(vertices, samples):

    vertices = np.asarray(
        vertices,
        dtype=float
    )

    closed = np.vstack([
        vertices,
        vertices[0]
    ])

    lengths = np.linalg.norm(
        np.diff(closed, axis=0),
        axis=1
    )

    cumulative = np.r_[
        0.0,
        np.cumsum(lengths)
    ]

    u = np.linspace(
        0.0,
        cumulative[-1],
        samples,
        endpoint=False
    )

    points = np.zeros(
        (samples, 2)
    )

    for i, s in enumerate(u):

        k = min(
            np.searchsorted(
                cumulative,
                s,
                side="right"
            ) - 1,
            len(lengths) - 1
        )

        alpha = (
            (s - cumulative[k]) / lengths[k]
            if lengths[k] > 0
            else 0.0
        )

        points[i] = (
            closed[k]
            + alpha * (
                closed[k + 1] - closed[k]
            )
        )

    return points


def shape_points(
    name,
    samples=401,
    scale=25.0,
    center=(175.0, 0.0)
):

    cx, cy = center

    name = name.lower()

    if name == "circle":

        angle = np.linspace(
            0.0,
            2.0 * np.pi,
            samples,
            endpoint=False
        )

        points = np.column_stack([
            scale * np.cos(angle),
            scale * np.sin(angle)
        ])

    elif name == "square":

        vertices = [
            [-scale, -scale],
            [scale, -scale],
            [scale, scale],
            [-scale, scale]
        ]

        points = resample_closed_polygon(
            vertices,
            samples
        )

    elif name == "star":

        angles = np.linspace(
            -np.pi / 2,
            3 * np.pi / 2,
            10,
            endpoint=False
        )

        radii = np.array([
            scale,
            scale * 0.42
        ] * 5)

        vertices = np.column_stack([
            radii * np.cos(angles),
            radii * np.sin(angles)
        ])

        points = resample_closed_polygon(
            vertices,
            samples
        )

    else:

        raise ValueError(
            f"Unknown shape: {name}"
        )

    points += np.array([
        cx,
        cy
    ])

    return points