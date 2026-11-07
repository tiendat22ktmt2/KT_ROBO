
import time
import numpy as np

from .common import (
    pose_error,
    weighted_system,
    finish_result,
    target_is_geometrically_reachable
)


def solve_analytical(
    robot,
    target,
    elbow=1,
    q_seed=None,
    cfg=None
):
    start = time.perf_counter()

    target = np.asarray(
        target,
        dtype=float
    ).reshape(-1)

    if target.size != 3:
        raise ValueError(
            "Target must contain [x, y, phi]."
        )

    if not np.all(np.isfinite(target)):
        raise ValueError(
            "Target must contain only finite values."
        )

    if elbow not in (-1, 1):
        raise ValueError(
            "Elbow must be either -1 or 1."
        )

    if q_seed is not None:
        q_seed = np.asarray(
            q_seed,
            dtype=float
        ).reshape(-1)

        if q_seed.size != 3:
            raise ValueError(
                "q_seed must contain 3 joint angles."
            )

        if not np.all(np.isfinite(q_seed)):
            raise ValueError(
                "q_seed must contain only finite values."
            )

    q_initial = (
        q_seed.copy()
        if q_seed is not None
        else np.zeros(3, dtype=float)
    )

    # Check geometric reachability
    if not target_is_geometrically_reachable(
        robot,
        target
    ):
        return finish_result(
            robot,
            target,
            q_initial,
            False,
            0,
            start,
            float("nan"),
            "geometrically_unreachable"
        )

    x, y, phi = target

    # Wrist position
    wx = x - robot.L3 * np.cos(phi)
    wy = y - robot.L3 * np.sin(phi)

    # Law of cosines
    c2 = (
        wx**2 + wy**2
        - robot.L1**2
        - robot.L2**2
    ) / (
        2 * robot.L1 * robot.L2
    )

    c2 = np.clip(c2, -1.0, 1.0)

    s2 = elbow * np.sqrt(
        max(0.0, 1.0 - c2**2)
    )

    q2 = np.arctan2(s2, c2)

    q1 = (
        np.arctan2(wy, wx)
        - np.arctan2(
            robot.L2 * s2,
            robot.L1 + robot.L2 * c2
        )
    )

    q3 = phi - q1 - q2

    q = np.array([
        q1,
        q2,
        q3
    ], dtype=float)

    # Choose equivalent angles close to initial seed
    if q_seed is not None:
        q = q_seed + (
            (q - q_seed + np.pi) % (2 * np.pi)
            - np.pi
        )

    # Validate the analytical solution using FK
    error = pose_error(
        target,
        robot.fk(q)
    )

    position_tolerance = (
        cfg.position_tolerance
        if cfg is not None
        else 1e-7
    )

    orientation_tolerance = (
        cfg.orientation_tolerance
        if cfg is not None
        else 1e-7
    )

    converged = (
        np.linalg.norm(error[:2])
        <= position_tolerance
        and abs(error[2])
        <= orientation_tolerance
    )

    # Calculate minimum singular value
    orientation_weight = (
        cfg.orientation_weight
        if cfg is not None
        else 40.0
    )

    Jw, _ = weighted_system(
        robot,
        q,
        np.zeros(3),
        orientation_weight
    )

    singular_values = np.linalg.svd(
        Jw,
        compute_uv=False
    )

    sigma_min = float(
        singular_values[-1]
    )

    return finish_result(
        robot,
        target,
        q,
        converged,
        1,
        start,
        sigma_min,
        "" if converged else "numerical_residual"
    )
