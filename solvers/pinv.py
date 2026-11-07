
import time
import numpy as np

from .common import (
    pose_error,
    weighted_system,
    finish_result,
    is_converged,
    limit_step,
    target_is_geometrically_reachable
)


def solve_pinv(
    robot,
    target,
    q0,
    cfg
):
    start = time.perf_counter()

    target = np.asarray(
        target,
        dtype=float
    ).reshape(-1)

    q = np.asarray(
        q0,
        dtype=float
    ).copy()

    if target.size != 3:
        raise ValueError(
            "Target must contain [x, y, phi]."
        )

    if not np.all(np.isfinite(target)):
        raise ValueError(
            "Target must contain only finite values."
        )

    if q.shape != (3,):
        raise ValueError(
            "q0 must contain 3 joint angles."
        )

    if not np.all(np.isfinite(q)):
        raise ValueError(
            "Initial joint angles must be finite."
        )

    sigma_min = float("nan")

    # Check geometric reachability before IK
    if not target_is_geometrically_reachable(
        robot,
        target
    ):
        return finish_result(
            robot,
            target,
            q,
            False,
            0,
            start,
            sigma_min,
            "geometrically_unreachable"
        )

    for it in range(cfg.max_iterations + 1):

        current = robot.fk(q)

        error = pose_error(
            target,
            current
        )

        if is_converged(error, cfg):
            return finish_result(
                robot,
                target,
                q,
                True,
                it,
                start,
                sigma_min
            )

        if it == cfg.max_iterations:
            break

        Jw, ew = weighted_system(
            robot,
            q,
            error,
            cfg.orientation_weight
        )

        singular_values = np.linalg.svd(
            Jw,
            compute_uv=False
        )

        sigma_min = float(
            singular_values[-1]
        )

        dq = (
            np.linalg.pinv(
                Jw,
                rcond=1e-10
            ) @ ew
        )

        dq = limit_step(
            dq,
            cfg.max_step_norm
        )

        q += cfg.step_gain * dq

    return finish_result(
        robot,
        target,
        q,
        False,
        cfg.max_iterations,
        start,
        sigma_min,
        "max_iterations"
    )
